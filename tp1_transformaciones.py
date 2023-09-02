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
    H2 = np.matmul(IMUTOC_T, H) # Transform
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


def plot_IMU_and_Camera(tp: pd.DataFrame):
    """
    Graphs the IMU and Cam0 paths
    """
    ax = plt.figure().add_subplot(projection='3d')
    # ax = plt.axes()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    ax.set_aspect('equal', 'box')
    plt.grid(visible=True, alpha=0.5)
    plt.plot(tp['x'], tp['y'], tp['z'])
    plt.show()
    # TODO


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
    plot_IMU_and_Camera(imu_dframe)

    # Save all to folder
    exit(1)  # TODO
    print(f"[[TO CAM0 SAVED TO {args.outpath}]]")
    print(f"[[NS TO S SAVED TO {args.outpath}]]")
    print(f"[[GRAPH SAVED TO {args.outpath}]]")
    print("¡DONE!")
