Tinyik
======

Tinyik is a simple and naive inverse kinematics solver.

It defines an actuator as a set of links and revolute joints from an origin. Here is an example of a robot arm that consists of two joints that rotate around z-axis and two links of 1.0 length along x-axis:

.. code-block:: python

    >>> import tinyik
    >>> arm = tinyik.Actuator(['z', [1., 0., 0.], 'z', [1., 0., 0.]])

Since the joint angles are zero by default, the end-effector position is at (2.0, 0, 0):

.. code-block:: python

    >>> arm.angles
    array([ 0.,  0.])
    >>> arm.ee
    array([ 2.,  0.,  0.])

Sets the joint angles to 30 and 60 degrees to calculate a new position of the end-effector:

.. code-block:: python

    >>> import numpy as np
    >>> arm.angles = [np.pi / 6, np.pi / 3]  # or np.deg2rad([30, 60])
    >>> arm.ee
    array([ 0.8660254,  1.5      ,  0.       ])

Sets a position of the end-effector to calculate the joint angles:

.. code-block:: python

    >>> arm.ee = [2 / np.sqrt(2), 2 / np.sqrt(2), 0.]
    >>> arm.angles
    array([  7.85398147e-01,   3.23715739e-08])
    >>> np.round(np.rad2deg(arm.angles))
    array([ 45.,   0.])


Installation
------------

.. code-block:: console

    $ pip install tinyik
