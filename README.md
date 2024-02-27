Author: Peter Krepkiy (pkrepkiy@umd.edu, krepkiyp@yahoo.com), 

Last edit: February 14, 2024

Revision: 0

Description:

This program reads .tif images for the purposes of extracting information
that is useful for research efforts with regard to the Long Solenoid
Experiment (LSE) at the University of Maryland Institute for Research in
Electronics and Applied Physics

Four inputs to the program are as follows:

Search Depth:
Change the depth of search for object. This value searches N pixels past
the first pixel with a value lower than the threshold. Increase for beam
images that are less dense or have more holes.

Reject Threshold X, Reject Threshold Y:
Define the threshold to reject objects that have less length
along X and Y than the reject value. Any object that has X or Y distance
less than N will be rejected. Increase this value if there is a lot of
bright FOD or artifacts.

Threshold multiplier:
Set the threshold multiplier. The algorithm searches for pixels that are
lower than this threshold to form the beam object. The formula is like so:
threshold = maxIntensity - maxIntensity*thresholdMultiplier
This value must be between 0 and 1.
