"""
Trabajo Practico N1 - Transformaciones
Integrantes:
 - Felipe Cinto
 - Guillermo Rolle
 - Nicolas Soncini
"""
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import transforms3d as t3d


# Types of the fields to read in from groun-truth file
FIELDNAMES_TYPES = {
    '#timestamp': np.int64,
    'p_RS_R_x [m]': np.float32,
    'p_RS_R_y [m]': np.float32,
    'p_RS_R_z [m]': np.float32,
    'q_RS_w []': np.float32,
    'q_RS_x []': np.float32,
    'q_RS_y []': np.float32,
    'q_RS_z []': np.float32
}


# Renaming of the fields to make it easier to work with
FIELDNAMES_RENAME = {
    '#timestamp': 'ts',
    'p_RS_R_x [m]': 'x',
    'p_RS_R_y [m]': 'y',
    'p_RS_R_z [m]': 'z',
    'q_RS_w []': 'qw',
    'q_RS_x []': 'qx',
    'q_RS_y []': 'qy',
    'q_RS_z []': 'qz'
}


#  Transformation from Cam0 to IMU (mav0/cam0/sensor.yaml)
CTOIMU_T = np.array([
    [0.0148655429818, -0.999880929698,   0.00414029679422, -0.0216401454975 ],
    [0.999557249008,   0.0149672133247,  0.025715529948,   -0.064676986768  ],
    [-0.0257744366974, 0.00375618835797, 0.999660727178,    0.00981073058949],
    [0.0,              0.0,              0.0,               1.0             ]
])
# IMUTOC_R = t3d.quaternions.mat2quat(IMUTOC_T[0:3:1,0:3:1]) # Rotation Matrix
# IMUTOC_t = IMUTOC_T[0:3,3:] # Translation Vector
# Transformation from IMU to Cam0
IMUTOC_T = np.linalg.inv(CTOIMU_T)


def transform_IMU_to_C(tp: pd.Series) -> pd.Series:
    """
    Transform IMU coordinates to Cam0 coordinates
    """
    R = t3d.quaternions.quat2mat(tp[['qw', 'qx', 'qy', 'qz']])
    t = np.array(tp[['x','y','z']])
    H = t3d.affines.compose(t, R, np.ones(3)) # Homogeneous
    H2 = np.matmul(H, CTOIMU_T) # Transform
    H2_t, H2_R, _, _ = t3d.affines.decompose(H2)
    tp[['x','y','z','qw','qx','qy','qz']] = np.concatenate([H2_t, t3d.quaternions.mat2quat(H2_R)]).ravel()
    return tp


def transform_Nanoseconds_to_Seconds(tp: pd.Series) -> pd.Series:
    """
    Transforms a timestamp in Nanoseconds of type integer to Seconds
    of type double
    """
    tp['ts'] = tp['ts'].astype(np.float64)/1e9
    return tp


def plot_IMU_and_Camera(imu_df: pd.DataFrame, cam_df: pd.DataFrame, outpath:Path, debug: bool = False):
    """
    Graphs the IMU and Cam0 paths
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    ax.set_aspect('equal')

    ax.set_title("Trayectorias de IMU y Cam0")

    # Plotear IMU
    color = next(ax._get_lines.prop_cycler)['color']
    imu_x = imu_df['x']
    imu_y = imu_df['y']
    imu_z = imu_df['z']
    ax.plot(imu_x, imu_y, imu_z, color=color, label='IMU', alpha=0.8)

    # Plotear Cam0
    # TODO
    color = next(ax._get_lines.prop_cycler)['color']
    cam0_x = cam_df['x']
    cam0_y = cam_df['y']
    cam0_z = cam_df['z']
    ax.plot(cam0_x, cam0_y, cam0_z, color=color, label='Cam0', alpha=0.8)

    def set_axes_equal(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax)

    plt.savefig(outpath / "ej5c.png", dpi=400)
    if debug:
        plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--ground-truth', required=True,
        help='CSV file with ground truth information'
    )
    parser.add_argument(
        '-o', '--outpath', type=Path, default='.',
        help='Output folder path to save data'
    )
    parser.add_argument(
        '--debug', action='store_true'
    )

    args = parser.parse_args()
    
    # Read CSV file, keep only relevant columns and rename to more useful names
    imu_dframe = pd.read_csv(args.ground_truth, skipinitialspace=True, dtype=FIELDNAMES_TYPES)
    imu_dframe.drop(imu_dframe.columns.difference(list(FIELDNAMES_TYPES.keys())), axis=1, inplace=True)
    imu_dframe.rename(columns=FIELDNAMES_RENAME, inplace=True)
    
    if args.debug:
        print(imu_dframe)

    # Print first 5 rows of original ground-truth and save to csv
    print("[[ORIGINAL]]")
    print(imu_dframe.head(6))
    imu_dframe.to_csv(args.outpath / 'original.csv', index=False)

    # Transform ground-truth coordinate system
    cam_dframe = imu_dframe.apply(transform_IMU_to_C, axis=1)
    cam_dframe['ts'] = cam_dframe['ts'].astype(np.int64)  # restore datatype

    # Print first 5 rows of transformed ground-truth and save to csv
    print("[[TO CAM0]]")
    print(cam_dframe.head(6))
    cam_dframe.to_csv(args.outpath / 'to_cam0.csv', index=False, float_format='%.6f')

    # Transform timestamps
    cam_dframe2 = cam_dframe.apply(transform_Nanoseconds_to_Seconds, axis=1)

    # Print first 5 rows of transformed ground-truth and save to csv
    print("[[NS TO S]]")
    print(cam_dframe2.head(6))
    cam_dframe.to_csv(args.outpath / 'to_cam0_sec.csv', index=False, float_format='%.6f')

    # Graph the original and transformed path
    print("[[GRAPH]]")
    plot_IMU_and_Camera(imu_dframe, cam_dframe, args.outpath, debug=args.debug)

    print("Done!")
