from skimage import io
from skimage import exposure
from skimage.transform import resize, radon, rescale
import numpy as np
from sklearn.impute import KNNImputer
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import torch
import torch.fft
import concurrent.futures


@torch.no_grad()
def denoise_tv_chambolle_torch(image, weight=0.1, n_iter_max=50):
    """
    Total Variation denoising using Chambolle's projection algorithm.
    """
    p = torch.zeros((2,) + image.shape, dtype=image.dtype, device=image.device)
    g = image
    tau = 0.25

    # Pre-allocate buffers to avoid repeated allocation in loop
    div_p = torch.zeros_like(image)
    grad_u_x = torch.zeros_like(image)
    grad_u_y = torch.zeros_like(image)

    for _ in range(n_iter_max):
        div_p.zero_()
        div_p[:-1, :] += p[0, :-1, :]
        div_p[1:, :] -= p[0, :-1, :]
        div_p[:, :-1] += p[1, :, :-1]
        div_p[:, 1:] -= p[1, :, :-1]

        u = g - weight * div_p

        grad_u_x.zero_()
        grad_u_y.zero_()
        grad_u_x[:-1, :] = u[1:, :] - u[:-1, :]
        grad_u_y[:, :-1] = u[:, 1:] - u[:, :-1]

        grad_mag = torch.hypot(grad_u_x, grad_u_y)

        denom = 1 + tau * grad_mag
        p[0] = (p[0] + tau * grad_u_x) / denom
        p[1] = (p[1] + tau * grad_u_y) / denom

    div_p.zero_()
    div_p[:-1, :] += p[0, :-1, :]
    div_p[1:, :] -= p[0, :-1, :]
    div_p[:, :-1] += p[1, :, :-1]
    div_p[:, 1:] -= p[1, :, :-1]

    return g - weight * div_p


class em_image:
    """
    A class for reading image file and removing stripe artifacts from SEM images.
    """

    def __init__(self, filename):
        """
        A class for reading image file and removing stripe artifacts from SEM images.

        Args:
            filename (tif,jpeg,png): SEM image file
        """
        self.original_image = io.imread(filename, as_gray=True)
        self.image = self.original_image.copy()
        self._update_spectral_props()
        self.wedge_size = []
        self.wedge_angle = []
        self.k_min = []
        self.reference_mask = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _update_spectral_props(self):
        self.dft = np.fft.fft2(self.image)
        self.fft_shifted = np.fft.fftshift(self.dft)

        # Precompute coordinates for mask generation to avoid re-allocation
        (nx, ny) = self.image.shape
        x = np.arange(-ny / 2, ny / 2, dtype=np.float32)
        y = np.arange(-nx / 2, nx / 2, dtype=np.float32)
        # Use broadcasting to avoid allocating full meshgrid arrays
        self.rr = np.square(x[None, :]) + np.square(y[:, None])
        self.phi = np.arctan2(y[:, None], x[None, :]) * -1

    def set_scale(self, shape=None):
        if shape is None:
            if self.image.shape != self.original_image.shape:
                self.image = self.original_image.copy()
                self._update_spectral_props()
        else:
            if self.image.shape != shape:
                self.image = resize(self.original_image, shape, anti_aliasing=True)
                self._update_spectral_props()

    def add_wedge(self, wedge_size, wedge_angle, k_min):
        """Function to add the parameters for defining the wedge.
           Multiple wedges can be added by invoking the function several times.

        Args:
            wedge_size (float): Angular range of the wedge in degrees.
            wedge_angle (float): Orientation of the wedge in degrees.
            k_min (float): Minimum frequency to start the wedge 1/px.
        """
        self.wedge_size.append(wedge_size)
        self.wedge_angle.append(wedge_angle)
        self.k_min.append(k_min)

    def detect_stripe_angle_radon(self, sigma=30, threshold=2.0):
        """
        Automated Real-Space Angle Detection using Radon Transform.
        1. Rescales image for speed.
        2. Removes cell body (High Pass Filter).
        3. Scans all angles to find the direction of maximum contrast.
        """
        theta, variances = self.compute_radon_variance(sigma)

        peaks_indices, _ = find_peaks(variances, height=np.std(variances) * threshold)

        if len(peaks_indices) > 0:
            refined_angles = theta[peaks_indices]
        else:
            refined_angles = np.array([theta[np.argmax(variances)]])

        # 5. Normalize to -90 to 90 (GUI format)
        # Radon 0 = Horizontal, Radon 90 = Vertical
        final_angles = 90 - refined_angles

        print(f"Detected Angles: {final_angles}")
        return final_angles

    def compute_radon_variance(self, sigma=30):
        print("Computing Radon Variance...")
        high_pass = self.get_radon_hpf(sigma)

        # Coarse Search (0 to 180 degrees)
        theta = np.linspace(0.0, 180.0, 180, endpoint=False)
        sinogram = radon(high_pass, theta=theta)

        # Find angle with max variance (sharpest projection)
        variances = np.var(sinogram, axis=0)
        return theta, variances

    def get_radon_hpf(self, sigma=30):
        scale = min(1.0, 512 / max(self.image.shape))
        if scale < 1.0:
            img_small = rescale(self.image, scale, anti_aliasing=True)
        else:
            img_small = self.image

        bg = gaussian_filter(img_small, sigma=sigma)
        high_pass = img_small - bg
        return high_pass

    def create_mask(self, wedgeSize, theta, kmin):
        """Function for creating mask for data deletion in the Fourier space

        Args:
            wedgeSize (float): Angular range of the wedge
            theta (float): Orientation of the wedge
            kmin (float): Minimum frequency of the wedge

        Returns:
            numpy.ndarray: An array of boolean values marking the region to be deleted
        """
        (nx, ny) = self.image.shape

        # Convert missing wedge size and theta to radians.
        rad_theta = -(theta + 90) * (np.pi / 180)
        dtheta = wedgeSize * (np.pi / 180)

        # Create the Mask
        mask = np.ones((nx, ny), dtype=bool)
        mask[
            (self.phi >= (rad_theta - dtheta / 2))
            & (self.phi <= (rad_theta + dtheta / 2))
        ] = 0
        mask[
            (self.phi >= (np.pi + rad_theta - dtheta / 2))
            & (self.phi <= (np.pi + rad_theta + dtheta / 2))
        ] = 0

        if theta + wedgeSize / 2 > 90:
            mask[self.phi >= (np.pi - dtheta / 2)] = 0

        mask[self.rr < np.square(kmin)] = 1  # Keep values below rmin.
        return mask

    def _compute_radial_threshold_map(
        self,
        reference_mask,
        bin_size=1,
        stat_type="median",
        factor=1.0,
        exclusion_radius=2,
    ):
        magnitudes = np.abs(self.fft_shifted)
        radius = np.sqrt(self.rr)

        flat_ref = reference_mask.ravel()
        flat_r = radius.ravel()
        flat_mag = magnitudes.ravel()

        valid_r = flat_r[flat_ref]
        valid_mag = flat_mag[flat_ref]

        bins = (valid_r / bin_size).astype(np.int32)
        max_bin_idx = int(np.max(radius) / bin_size) + 1
        lookup_table = np.zeros(max_bin_idx, dtype=np.float32)

        if stat_type == "mean_std":
            # Vectorized calculation using bincount (much faster than loop)
            counts = np.bincount(bins, minlength=max_bin_idx)
            sums = np.bincount(bins, weights=valid_mag, minlength=max_bin_idx)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

            sums_sq = np.bincount(bins, weights=valid_mag**2, minlength=max_bin_idx)
            means_sq = np.divide(
                sums_sq, counts, out=np.zeros_like(sums_sq), where=counts > 0
            )
            stds = np.sqrt(np.maximum(means_sq - means**2, 0))

            lookup_table = means + (stds * factor)

            # Handle single-item bins (std=0) to match original logic: val * 2
            mask_single = counts == 1
            if np.any(mask_single):
                lookup_table[mask_single] = means[mask_single] * 2
        else:
            # Original logic for median (harder to vectorize without sorting)
            sort_idx = np.argsort(bins)
            sorted_bins = bins[sort_idx]
            sorted_mags = valid_mag[sort_idx]
            unique_bins, bin_start_indices = np.unique(sorted_bins, return_index=True)
            mags_split = np.split(sorted_mags, bin_start_indices[1:])

            for b, mags in zip(unique_bins, mags_split):
                if b < max_bin_idx and mags.size > 0:
                    if stat_type == "median":
                        val = np.median(mags)
                        lookup_table[b] = val * factor

        for b in range(1, max_bin_idx):
            if lookup_table[b] == 0:
                lookup_table[b] = lookup_table[b - 1]

        pixel_bins = (radius / bin_size).astype(np.int32)
        pixel_bins = np.clip(pixel_bins, 0, max_bin_idx - 1)
        threshold_map = lookup_table[pixel_bins]

        threshold_map[radius < exclusion_radius] = np.inf

        return threshold_map

    def get_combined_mask(self):
        scale_factor = self.image.shape[0] / self.original_image.shape[0]
        combined_mask = np.ones(self.image.shape, dtype=bool)
        for size, angle, kmin in zip(self.wedge_size, self.wedge_angle, self.k_min):
            m = self.create_mask(size, angle, kmin * scale_factor)
            combined_mask = combined_mask & m
        return combined_mask

    def process_image(
        self,
        method="knn",
        tv_iter=100,
        tv_weight=0.1,
        safety_factor=1.1,
        remove_wedge=False,
        callback=None,
        cancel_event=None,
    ):
        """
        Removes data from the masked regions and fills the missing regions using KNNImputation
        or TV Reconstruction.

        Args:
            method (str): 'knn', 'tv', 'pocs', 'weighted_l1', 'zero'.
            tv_iter (int): Number of iterations for TV reconstruction.
            tv_weight (float): Denoising weight for TV reconstruction.
            safety_factor (float): Safety factor for thresholding methods.
            remove_wedge (bool): Whether to zero out wedge region in POCS+L1.
            callback (func): Optional callback function for progress updates (0-100).
            cancel_event (threading.Event): Event to signal cancellation.
        """
        scale_factor = self.image.shape[0] / self.original_image.shape[0]

        # Common: Combine masks from all wedges
        combined_mask = self.get_combined_mask()

        # Pre-calculate unshifted mask for FFT operations
        combined_mask_unshifted = np.fft.ifftshift(combined_mask)

        with torch.no_grad():
            match method:
                case "tv":
                    self._reconstruct_tv(
                        combined_mask_unshifted, tv_iter, tv_weight, callback, cancel_event
                    )
                case "pocs":
                    self._reconstruct_pocs(
                        combined_mask_unshifted, tv_iter, callback, cancel_event
                    )
                case "weighted_l1":
                    self._reconstruct_weighted_l1(
                        combined_mask, tv_iter, tv_weight, callback, cancel_event
                    )
                case "dual_domain":
                    self._reconstruct_dual_domain(
                        combined_mask,
                        combined_mask_unshifted,
                        tv_iter,
                        tv_weight,
                        safety_factor,
                        callback,
                        cancel_event,
                    )
                case "pocs_l1":
                    self._reconstruct_pocs_l1(
                        combined_mask,
                        combined_mask_unshifted,
                        tv_iter,
                        safety_factor,
                        callback,
                        cancel_event,
                        remove_wedge=remove_wedge,
                    )
                case "fista":
                    self._reconstruct_fista(
                        combined_mask, tv_iter, safety_factor, callback, cancel_event
                    )
                case "zero":
                    self._reconstruct_zero(combined_mask)
                case "fista_pocs":
                    self._reconstruct_fista_pocs(
                        combined_mask, tv_iter, safety_factor, callback, cancel_event
                    )
                case _:
                    self._reconstruct_knn(scale_factor, callback, cancel_event)

    def _reconstruct_tv(self, mask_unshifted, n_iter, weight, callback, cancel_event):
        # Initial estimate: IFFT of known data (zeros elsewhere)
        dft_torch = torch.from_numpy(self.dft).to(self.device)
        mask_unshifted_torch = torch.from_numpy(mask_unshifted).to(self.device)

        f_input = dft_torch.clone()
        f_input[~mask_unshifted_torch] = 0
        img_est = torch.fft.ifft2(f_input).real

        for i in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled")
            img_est = denoise_tv_chambolle_torch(img_est, weight=weight)
            f_est = torch.fft.fft2(img_est)
            f_est[mask_unshifted_torch] = dft_torch[mask_unshifted_torch]
            img_est = torch.fft.ifft2(f_est).real

            if callback and i % 2 == 0:
                callback(int((i + 1) / n_iter * 100))

        self.processed_fft = np.fft.fftshift(f_est.cpu().numpy())

    def _reconstruct_pocs(self, mask_unshifted, n_iter, callback, cancel_event):
        dft_torch = torch.from_numpy(self.dft).to(self.device)
        mask_unshifted_torch = torch.from_numpy(mask_unshifted).to(self.device)

        f_est = dft_torch.clone()
        f_est[~mask_unshifted_torch] = 0

        prev_img_est = None
        tol = 1e-6

        for i in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled")
            img_est = torch.fft.ifft2(f_est).real
            img_est[img_est < 0] = 0

            if prev_img_est is not None:
                diff = torch.norm(img_est - prev_img_est) / (
                    torch.norm(prev_img_est) + 1e-8
                )
                if diff < tol:
                    if callback:
                        callback(100)
                    break
            prev_img_est = img_est.clone()

            f_est = torch.fft.fft2(img_est)
            f_est[mask_unshifted_torch] = dft_torch[mask_unshifted_torch]

            if callback and i % 5 == 0:
                callback(int((i + 1) / n_iter * 100))

        self.processed_fft = np.fft.fftshift(f_est.cpu().numpy())

    def _reconstruct_weighted_l1(
        self, combined_mask, n_iter, tv_weight, callback, cancel_event
    ):
        # Setup on CPU
        artifact_mask = ~combined_mask
        reference_mask = combined_mask

        lambda_map = self._compute_radial_threshold_map(
            reference_mask,
            bin_size=2,
            stat_type="mean_std",
            factor=tv_weight,
            exclusion_radius=8,
        )

        # Move to GPU
        f_est = torch.from_numpy(self.fft_shifted).to(self.device)
        artifact_mask_torch = torch.from_numpy(artifact_mask).to(self.device)
        lambda_map_torch = torch.from_numpy(lambda_map).to(self.device)

        for i in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled")
            curr_mag = torch.abs(f_est)
            curr_phase = torch.angle(f_est)

            is_outlier = (curr_mag > lambda_map_torch) & artifact_mask_torch

            if torch.any(is_outlier):
                excess = curr_mag[is_outlier] - lambda_map_torch[is_outlier]
                damping_factor = 0.5
                new_mag = curr_mag[is_outlier] - (excess * damping_factor)
                f_est[is_outlier] = new_mag * torch.exp(1j * curr_phase[is_outlier])

            if callback and i % 5 == 0:
                callback(int((i + 1) / n_iter * 100))

        self.processed_fft = f_est.cpu().numpy()

    def _reconstruct_dual_domain(
        self,
        combined_mask,
        mask_unshifted,
        n_iter,
        tv_weight,
        safety_factor,
        callback,
        cancel_event,
    ):
        # Setup on CPU
        trust_mask = mask_unshifted
        artifact_mask = ~mask_unshifted

        ref_mask_shifted = combined_mask
        self.reference_mask = ref_mask_shifted

        threshold_map = self._compute_radial_threshold_map(
            ref_mask_shifted,
            bin_size=2,
            stat_type="mean_std",
            factor=safety_factor,
            exclusion_radius=8,
        )

        threshold_map_unshifted = np.fft.ifftshift(threshold_map)

        # Move to GPU
        f_est = torch.from_numpy(self.dft).to(self.device)
        dft_torch = torch.from_numpy(self.dft).to(self.device)
        trust_mask_torch = torch.from_numpy(trust_mask).to(self.device)
        artifact_mask_torch = torch.from_numpy(artifact_mask).to(self.device)
        threshold_map_unshifted_torch = torch.from_numpy(threshold_map_unshifted).to(
            self.device
        )

        for i in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled")
            img_curr = torch.fft.ifft2(f_est).real
            img_smooth = denoise_tv_chambolle_torch(img_curr, weight=tv_weight)
            f_smooth = torch.fft.fft2(img_smooth)
            f_smooth[trust_mask_torch] = dft_torch[trust_mask_torch]

            curr_mags = torch.abs(f_smooth)
            curr_phases = torch.angle(f_smooth)

            outliers = (curr_mags > threshold_map_unshifted_torch) & artifact_mask_torch

            if torch.any(outliers):
                excess = curr_mags[outliers] - threshold_map_unshifted_torch[outliers]
                new_mags = curr_mags[outliers] - (excess * 0.5)
                f_smooth[outliers] = new_mags * torch.exp(1j * curr_phases[outliers])

            f_est = f_smooth

            if callback and i % 2 == 0:
                callback(int((i + 1) / n_iter * 100))

        self.processed_fft = np.fft.fftshift(f_est.cpu().numpy())

    def _reconstruct_pocs_l1(
        self,
        combined_mask,
        mask_unshifted,
        n_iter,
        safety_factor,
        callback,
        cancel_event,
        remove_wedge=False,
    ):
        # Setup on CPU
        trust_mask = mask_unshifted
        artifact_mask = ~mask_unshifted
        ref_mask_shifted = combined_mask

        artifact_mask_shifted = ~combined_mask
        soft_mask_shifted = gaussian_filter(
            artifact_mask_shifted.astype(float), sigma=2
        )
        soft_mask = np.fft.ifftshift(soft_mask_shifted)

        threshold_map = self._compute_radial_threshold_map(
            ref_mask_shifted,
            bin_size=2,
            stat_type="median",
            factor=safety_factor,
            exclusion_radius=2,
        )

        threshold_map_unshifted = np.fft.ifftshift(threshold_map)

        # Move to GPU
        f_est = torch.from_numpy(self.dft).to(self.device)
        dft_torch = torch.from_numpy(self.dft).to(self.device)
        trust_mask_torch = torch.from_numpy(trust_mask).to(self.device)
        artifact_mask_torch = torch.from_numpy(artifact_mask).to(self.device)
        threshold_map_unshifted_torch = torch.from_numpy(threshold_map_unshifted).to(
            self.device
        )
        soft_mask_torch = torch.from_numpy(soft_mask).to(self.device)

        if remove_wedge:
            f_est[artifact_mask_torch] = 0
        print(f"Starting Iterative Removal (Safety: {safety_factor})...")

        prev_img = None
        x = 0
        i = 0

        for i in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled")
            img_est = torch.fft.ifft2(f_est).real
            img_est[img_est < 0] = 0

            f_est = torch.fft.fft2(img_est)
            f_est[trust_mask_torch] = dft_torch[trust_mask_torch]

            curr_mags = torch.abs(f_est)
            curr_phases = torch.angle(f_est)

            outliers = (curr_mags > threshold_map_unshifted_torch) & artifact_mask_torch

            if i == 0:
                print(f"Iter 0: Detected {torch.sum(outliers)} stripe pixels.")
            x += torch.sum(outliers)
            if torch.any(outliers):
                excess = curr_mags[outliers] - threshold_map_unshifted_torch[outliers]
                damping = 0.8 * excess * soft_mask_torch[outliers]
                new_mags = curr_mags[outliers] - damping
                f_est[outliers] = new_mags * torch.exp(1j * curr_phases[outliers])

            if callback and i % 5 == 0:
                callback(int((i + 1) / n_iter * 100))
        print(f"Detected {x} stripe pixels in {i} iterations ")
        self.processed_fft = np.fft.fftshift(f_est.cpu().numpy())

    def _reconstruct_fista(
        self, combined_mask, n_iter, safety_factor, callback, cancel_event
    ):
        """
        FISTA-accelerated reconstruction.
        Converges significantly faster than standard POCS-L1.
        """
        # 1. Setup Constraints (Same as POCS-L1)
        trust_mask = np.fft.ifftshift(combined_mask)
        artifact_mask = ~trust_mask

        # Build Threshold Map
        threshold_map = self._compute_radial_threshold_map(
            combined_mask,
            bin_size=2,
            stat_type="median",
            factor=safety_factor,
            exclusion_radius=2,
        )
        threshold_map_unshifted = np.fft.ifftshift(threshold_map)

        # Move to GPU
        device = self.device

        # Initialize Variables
        # x_k: The current estimate
        # y_k: The "Momentum" point (where we think we are going)
        x_k = torch.from_numpy(self.dft).to(device)
        y_k = x_k.clone()
        t_k = 1.0  # Momentum scalar

        dft_original = x_k.clone()
        mask_good = torch.from_numpy(trust_mask).to(device)
        mask_bad = torch.from_numpy(artifact_mask).to(device)
        thresh_map = torch.from_numpy(threshold_map_unshifted).to(device)

        print(f"Starting FISTA-L1 (Safety: {safety_factor})...")

        for k in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled")

            # --- 1. Gradient Step (Data Consistency) ---
            y_k[mask_good] = dft_original[mask_good]

            # --- 2. Proximal Step (L1 Shrinkage) ---
            curr_mags = torch.abs(y_k)
            phases = torch.angle(y_k)

            outliers = (curr_mags > thresh_map) & mask_bad

            if torch.any(outliers):
                excess = curr_mags[outliers] - thresh_map[outliers]
                new_mags = curr_mags[outliers] - (excess * 0.8)
                y_k[outliers] = new_mags * torch.exp(1j * phases[outliers])

            # --- 3. Positivity Constraint (POCS) ---
            img_est = torch.fft.ifft2(y_k).real
            img_est = torch.clamp(img_est, min=0)
            x_next = torch.fft.fft2(img_est)

            # --- 4. FISTA MOMENTUM UPDATE ---
            t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            alpha = (t_k - 1) / t_next
            y_k = x_next + alpha * (x_next - x_k)

            x_k = x_next
            t_k = t_next

            if callback and k % 5 == 0:
                callback(int((k + 1) / n_iter * 100))

        self.processed_fft = np.fft.fftshift(x_k.cpu().numpy())

    def _reconstruct_fista_pocs(
        self, combined_mask, n_iter, safety_factor, callback, cancel_event
    ):
        """
        FISTA-POCS: Combines Nesterov Acceleration (Speed) with
        Positivity Constraints (Physics).
        """
        # 1. Setup Constraints
        trust_mask = np.fft.ifftshift(combined_mask)
        artifact_mask = ~trust_mask

        # Build Threshold Map (Recycle your helper logic)
        # (Assuming you have a helper or copy-paste the radial median logic here)
        # For this example, I'll use a placeholder for the map logic
        threshold_map = self._compute_radial_threshold_map(
            combined_mask,
            bin_size=2,
            stat_type="median",
            factor=safety_factor,
            exclusion_radius=2,
        )
        threshold_map_unshifted = np.fft.ifftshift(threshold_map)

        # Move to GPU
        import torch

        device = self.device

        # Initialize Variables
        # x_k: Current Estimate
        # y_k: Momentum Point
        # t_k: Momentum Scalar
        x_k = torch.from_numpy(self.dft).to(device)
        y_k = x_k.clone()
        t_k = 1.0

        dft_original = x_k.clone()
        mask_good = torch.from_numpy(trust_mask).to(device)
        mask_bad = torch.from_numpy(artifact_mask).to(device)
        thresh_map = torch.from_numpy(threshold_map_unshifted).to(device)

        print(f"Starting FISTA-POCS (Safety: {safety_factor})...")

        for k in range(n_iter):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            # --- Step 1: Gradient Descent (Data Consistency) ---
            # We enforce consistency on the MOMENTUM point y_k
            y_k[mask_good] = dft_original[mask_good]

            # --- Step 2: Proximal Operator (L1 Shrinkage) ---
            # We clean the stripes from y_k
            curr_mags = torch.abs(y_k)
            phases = torch.angle(y_k)

            outliers = (curr_mags > thresh_map) & mask_bad

            if torch.any(outliers):
                excess = curr_mags[outliers] - thresh_map[outliers]
                new_mags = curr_mags[outliers] - (excess * 0.8)
                y_k[outliers] = new_mags * torch.exp(1j * phases[outliers])

            # --- Step 3: POCS Projection (Positivity) ---
            # CRITICAL ADDITION: Project to Real Space -> Clamp -> Back to Fourier
            img_est = torch.fft.ifft2(y_k).real

            # The Constraint: "Mass cannot be negative"
            img_est = torch.clamp(img_est, min=0)

            # The new estimate x_{k+1}
            x_next = torch.fft.fft2(img_est)

            # --- Step 4: FISTA Momentum Update ---
            # Calculate new momentum scalar
            t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2

            # Calculate interpolation weight
            alpha = (t_k - 1) / t_next

            # Update y_{k+1} (Look ahead)
            # We push the momentum vector forward based on the purified, positive estimate
            y_k = x_next + alpha * (x_next - x_k)

            # Update trackers
            x_k = x_next
            t_k = t_next

            if callback and k % 5 == 0:
                callback(int((k + 1) / n_iter * 100))

        self.processed_fft = np.fft.fftshift(x_k.cpu().numpy())

    def _reconstruct_zero(self, combined_mask):
        f_est = torch.from_numpy(self.fft_shifted).to(self.device)
        mask = torch.from_numpy(combined_mask).to(self.device)
        f_est[~mask] = 0
        self.processed_fft = f_est.cpu().numpy()

    def _process_wedge_knn(self, args):
        size, angle, kmin, scale_factor, base_fft = args
        knn_imputer = KNNImputer(missing_values=0, n_neighbors=1, weights="distance")

        mask = self.create_mask(size, angle, kmin * scale_factor)
        temp_array = base_fft.copy()

        if angle > 45 or angle < -45:
            temp_array[~mask] = 0
            k_impute_real = knn_imputer.fit_transform(temp_array.real.T)
            k_impute_imag = knn_imputer.fit_transform(temp_array.imag.T)
            c_k = np.empty(base_fft.shape, dtype=np.complex128)
            c_k.real = k_impute_real.T
            c_k.imag = k_impute_imag.T
        else:
            temp_array[~mask] = 0
            k_impute_real = knn_imputer.fit_transform(temp_array.real)
            k_impute_imag = knn_imputer.fit_transform(temp_array.imag)
            c_k = np.empty(base_fft.shape, dtype=np.complex128)
            c_k.real = k_impute_real
            c_k.imag = k_impute_imag

        return c_k, mask

    def _reconstruct_knn(self, scale_factor, callback, cancel_event):
        final_fft = self.fft_shifted.copy()
        wedges = [
            (size, angle, kmin, scale_factor, self.fft_shifted)
            for size, angle, kmin in zip(self.wedge_size, self.wedge_angle, self.k_min)
        ]
        total_wedges = len(wedges)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_wedge_knn, args) for args in wedges
            ]

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                if cancel_event and cancel_event.is_set():
                    executor.shutdown(wait=False)
                    raise InterruptedError("Processing cancelled")

                try:
                    filled_fft, mask = future.result()
                    final_fft[~mask] = filled_fft[~mask]
                except Exception as e:
                    print(f"Error in KNN thread: {e}")
                    raise e

                if callback:
                    callback(int((idx + 1) / total_wedges * 100))

        self.processed_fft = final_fft

    def reconstruct_image(self):
        """
        Reconstructs the processed imaged via inverse Fourier Transformation
        """
        f_ishift = np.fft.ifftshift(self.processed_fft)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        normalized_image = (img_back - np.min(img_back)) * (
            255.0 / (np.max(img_back) - np.min(img_back))
        )
        self.img_recon = normalized_image.astype("uint8")

        img_in_norm = (self.image - np.min(self.image)) * (
            255.0 / (np.max(self.image) - np.min(self.image))
        )
        self.img_diff = img_in_norm - normalized_image

    def recover_texture(self, sigma_clip=3.0):
        """
        Extracts safe, non-stripe texture from the residual and adds it back.
        """
        if not hasattr(self, "img_recon"):
            return

        cleaned_image = self.img_recon.astype(float)

        # Handle Quick Processing case where img_recon is smaller than current self.image
        current_image = self.image
        if self.image.shape != cleaned_image.shape:
            current_image = resize(self.image, cleaned_image.shape, anti_aliasing=True)

        # Normalize input image to match cleaned_image (0-255)
        img_min = np.min(current_image)
        img_max = np.max(current_image)
        if img_max - img_min == 0:
            img_in_norm = np.zeros_like(current_image)
        else:
            img_in_norm = (current_image - img_min) * (255.0 / (img_max - img_min))

        # 1. Calculate the Residual (What was removed)
        residual = img_in_norm - cleaned_image

        # 2. Move to Fourier Space
        f_res = np.fft.fftshift(np.fft.fft2(residual))

        # 3. Identify the "Stripe Signal" vs "Texture Signal"
        magnitudes = np.abs(f_res)
        mean_val = np.mean(magnitudes)
        std_val = np.std(magnitudes)

        # Define a "Ceiling" for allowed texture
        limit = mean_val + (std_val * sigma_clip)

        # 4. Clip the Residual
        mask_stripes = magnitudes > limit

        # Soft clipping
        f_res[mask_stripes] = (f_res[mask_stripes] / magnitudes[mask_stripes]) * limit

        # 5. Reconstruct the "Safe Texture"
        texture_only = np.real(np.fft.ifft2(np.fft.ifftshift(f_res)))

        # 6. Add it back!
        self.img_texture = cleaned_image + texture_only
        self.img_texture = np.clip(self.img_texture, 0, 255).astype("uint8")

    def enhance_contrast(self, p1=1, p2=90):
        """A simple contrast enhancer

        Args:
            p1 (int, optional): Lower percentile. Defaults to 1.
            p2 (int, optional): Higher percentile. Defaults to 90.
        """
        p_l, p_h = np.percentile(self.img_recon, (p1, p2))
        self.img_enhanced = exposure.rescale_intensity(
            self.img_recon, in_range=(p_l, p_h)
        )

    def clear_mask(self):
        self.wedge_size = []
        self.wedge_angle = []
        self.k_min = []

    def delete_mask(self, index):
        del self.wedge_size[index]
        del self.wedge_angle[index]
        del self.k_min[index]
