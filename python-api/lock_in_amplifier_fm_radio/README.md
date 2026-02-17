# FM Radio with Moku and GNU Radio

This example contains control code for streaming IQ samples from Moku Lock-in Amplifier. It also includes necessary GNU Radio files to implement a simple FM Radio Receiver.

Python script for Moku streaming has standard dependancies similar to other examples. To run GNU Radio side of this example, the user must first install [GNU Radio](https://wiki.gnuradio.org/index.php/InstallingGR) and [DIFI OOT module](https://github.com/DIFI-Consortium/gr-difi) respectively.

After GNU Radio setup is complete, the user must run the GNU Radio flowgraph to listen to the specified radio channel, while streaming script is responsible for sending IQ samples exracted from Moku hardware as DIFI packets.



