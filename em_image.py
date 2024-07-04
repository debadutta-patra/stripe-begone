from skimage import io
from skimage import exposure
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer


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
        self.image = io.imread(filename, as_gray=True)
        self.dft = np.fft.fft2(self.image)
        self.fft_shifted = np.fft.fftshift(self.dft)
        self.wedge_size = []
        self.wedge_angle = []
        self.k_min = []

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
        rad_theta = -(theta+90)*(np.pi/180)
        dtheta = wedgeSize*(np.pi/180)

        # Create coordinate grid in polar
        x = np.arange(-nx/2, nx/2-1, dtype=np.float64)
        y = np.arange(-ny/2, ny/2-1, dtype=np.float64)
        [x, y] = np.meshgrid(x, y, indexing='xy')
        rr = (np.square(x) + np.square(y))
        phi = np.arctan2(y, x)
        phi *= -1

        # Create the Mask
        mask = np.ones((ny, nx), dtype=np.int8)
        mask[np.where((phi >= (rad_theta-dtheta/2)) &
                      (phi <= (rad_theta+dtheta/2)))] = 0
        mask[np.where((phi >= (np.pi+rad_theta-dtheta/2)) &
                      (phi <= (np.pi+rad_theta+dtheta/2)))] = 0

        if theta + wedgeSize/2 > 90:
            mask[np.where(phi >= (np.pi - dtheta/2))] = 0

        mask[np.where(rr < np.square(kmin))] = 1  # Keep values below rmin.
        mask = np.array(mask, dtype=bool)
        mask = np.transpose(mask)
        return mask

    def process_image(self):
        """
            Removes data from the masked regions and fills the missing regions using KNNImputation
        """
        temp_array = self.fft_shifted.copy()
        knn_imputer = KNNImputer(
            missing_values=0, n_neighbors=1, weights='distance')
        for size, angle, kmin in zip(self.wedge_size, self.wedge_angle, self.k_min):
            mask = self.create_mask(size, angle, kmin)
            if angle > 45 or angle < -45:
                temp_array[~mask] = 0
                k_impute_real = knn_imputer.fit_transform(temp_array.real.T)
                k_impute_imag = knn_imputer.fit_transform(temp_array.imag.T)
                c_k = np.empty(self.image.shape, dtype=np.complex128)
                c_k.real = k_impute_real.T
                c_k.imag = k_impute_imag.T
                temp_array = c_k
            else:
                temp_array[~mask] = 0
                k_impute_real = knn_imputer.fit_transform(temp_array.real)
                k_impute_imag = knn_imputer.fit_transform(temp_array.imag)
                c_k = np.empty(self.image.shape, dtype=np.complex128)
                c_k.real = k_impute_real
                c_k.imag = k_impute_imag
                temp_array = c_k

        self.processed_fft = temp_array

    def reconstruct_image(self):
        """
            Reconstructs the processed imaged via inverse Fourier Transformation
        """
        f_ishift = np.fft.ifftshift(self.processed_fft)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        normalized_image = (img_back - np.min(img_back)) * \
            (255.0 / (np.max(img_back) - np.min(img_back)))
        self.img_recon = normalized_image.astype('uint8')

    def enhance_contrast(self, p1=1, p2=90):
        """A simple contrast enhancer

        Args:
            p1 (int, optional): Lower percentile. Defaults to 1.
            p2 (int, optional): Higher percentile. Defaults to 90.
        """
        p_l, p_h = np.percentile(self.img_recon, (p1, p2))
        self.img_enhanced = exposure.rescale_intensity(
            self.img_recon, in_range=(p_l, p_h))

    def clear_mask(self):
        self.wedge_size = []
        self.wedge_angle = []
        self.k_min = []

    def delete_mask(self, index):
        del self.wedge_size[index]
        del self.wedge_angle[index]
        del self.k_min[index]
