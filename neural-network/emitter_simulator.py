"""

Author: Tranter Tech
Date: 2024
"""

import numpy as np


class QuantumEmitter:
    """
    Class for simulating a quantum emitter with a Gaussian beam profile.

    This simulator is used in the Emitter_control.ipynb example
    """

    def __init__(self, wavelength, waist):
        self.wavelength = wavelength
        self.w0 = waist
        self.zr = np.pi * self.w0**2 / self.wavelength

        self.X = None
        self.Y = None
        self.angles = [np.pi / 2, np.pi / 2]
        self.e_target = None
        self.e_current = None

    def get_field(self, x, y, z, mods):
        """
        Simulate the field at a given point with modifications due to the angle and offset
        :param x: x position
        :param y: y position
        :param z: z position
        :param mods: modifications in the form of [x translation, y translation, x scale, y scale]
        :return: the field at those points
        """
        return self.E((x + mods[0]) * mods[2], (y + mods[1]) * mods[3], z)

    def update_current(self, mods):
        """
        Update the current incident field for a given set of modifications
        :param mods: the list of modifications to provide to self.get_field
        :return: None
        """
        self.e_current = np.abs(self.get_field(self.X, self.Y, 1e-10, mods)) ** 2

    def set_XY(self, X, Y):
        """
        Set the X and Y arrays that correspond to the simulation domain, ij formatting from np.meshgrid is expected.
        :param X: X-array of points (1D)
        :param Y: Y-array of points (1D)
        :return:
        """
        self.X = X
        self.Y = Y
        self.e_target = self.get_field(
            X, Y, 1e-10, (0, 0, self.new_scale(self.angles[0]), self.new_scale(self.angles[1]))
        )
        self.e_target = np.abs(self.e_target) ** 2

    def get_overlap(self):
        # return the approximate overlap of the two beams
        return (self.e_target * self.e_current).sum() / (self.e_target**2).sum()

    def get_counts(self):
        # turn the overlap into a count value, clip it to avoid the approximation error
        # assume we're returning a 16-bit integer
        return np.clip(int(self.get_overlap() * 2**16), 0, 2**16)

    def time_step(self, offsets, shears, angles):
        """
        Perform one time step by setting the offsets and shear/angle of the beam. Modulations performed in each axis
        will form the signal that we use for correction.
        :param offsets: offsets to apply to the x and y axis, expect an iterable of size 2
        :param shears: shears to apply to the x and y axis, expect an iterable of size 2
        :param angles: new angles for the z-x and z-y angles, expect an iterable of size 2
        :return: the difference in the counts for each modulation
        """

        # unpack our new positions
        mods = (*offsets, *shears)
        self.angles = angles
        self.update_current(mods)

        # find the initial counts
        base_count = self.get_counts()

        # x offset
        new_mod = [m for m in mods]
        new_mod[0] += 1e-7
        self.update_current(new_mod)
        x_offset = self.get_counts()

        # y offset
        new_mod = [m for m in mods]
        new_mod[1] += 1e-7
        self.update_current(new_mod)
        y_offset = self.get_counts()

        # x angle
        new_mod = [m for m in mods]
        new_mod[2] = self.new_scale(self.angles[0] - np.pi / 64)
        self.update_current(new_mod)
        x_angle = self.get_counts()

        # y angle
        new_mod = [m for m in mods]
        new_mod[3] = self.new_scale(self.angles[1] - np.pi / 64)
        self.update_current(new_mod)
        y_angle = self.get_counts()

        diff_counts = np.array([x_offset, y_offset, x_angle, y_angle]) - base_count
        self.update_current(mods)

        return diff_counts

    def w(self, z):
        """
        Find the waist for a given z-position along the beam axis
        :param z: position along the beam axis in m
        :return: the beam waist in m
        """
        return self.w0 * np.sqrt(1 + (z / self.zr) ** 2)

    def E(self, x, y, z):
        """
        Calculate the electric field at given point in space
        :param x: x position in m
        :param y: y position in m
        :param z: z position in m
        :return:
        """
        E0 = 1  # amplitude, set to 1 for simplicity
        k = np.pi * 2 / self.wavelength  # the k-vector or wave vector
        Rz = z * (1 + (self.zr / z) ** 2)  # the radius of curvature

        # the three terms of the calculation that are multiplied together
        t1 = E0 * (self.w0 / self.w(z))
        t2 = np.exp(-(x**2 + y**2) / self.w(z) ** 2)
        t3 = np.exp(1j * (k * z - np.arctan(z / self.zr) + (k * (x**2 + y**2)) / (2 * Rz)))

        # return the complex electric field
        return t1 * t2 * t3

    def new_length(self, w, theta):
        """
        Function for calculating the shear projection of the beam
        :param w: the beam waist, set = 1 to get the factor
        :param theta: the angle between the normal z axis and other relevant axis (x or y)
        :return: the factor by which to distort the beam
        """
        eps = np.pi / 2 - theta
        x = 2 * np.sin(eps) / w
        return 2 * np.sqrt(x**2 + (w / 2) ** 2)

    def new_scale(self, theta):
        """
        Convenience function for returning the new scale of the beam in the relevant axis
        :param theta: angle between the axis and z-axis
        :return: the scale factor
        """
        return self.new_length(1, theta)
