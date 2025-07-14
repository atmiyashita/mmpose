# ===== Silkworm-11 keypoints, COCO format =====
dataset_info = dict(
    dataset_name='silkworm11',
    paper_info=dict(
        title='Silkworm Pose Estimation Dataset',
        year='2025',
        homepage='',
    ),
    keypoint_info={
        0:  dict(name='H1', id=0,  color=[255, 0, 0], type='upper', swap=''),
        1:  dict(name='T1', id=1,  color=[255, 0, 0], type='upper', swap=''),
        2:  dict(name='T2', id=2,  color=[255, 0, 0], type='upper', swap=''),
        3:  dict(name='T3', id=3,  color=[255, 0, 0], type='upper', swap=''),
        4:  dict(name='A2', id=4,  color=[255, 0, 0], type='lower', swap=''),
        5:  dict(name='A3', id=5,  color=[255, 0, 0], type='lower', swap=''),
        6:  dict(name='A4', id=6,  color=[255, 0, 0], type='lower', swap=''),
        7:  dict(name='A5', id=7,  color=[255, 0, 0], type='lower', swap=''),
        8:  dict(name='A6', id=8,  color=[255, 0, 0], type='lower', swap=''),
        9:  dict(name='A8', id=9,  color=[255, 0, 0], type='lower', swap=''),
        10: dict(name='A9', id=10, color=[255, 0, 0], type='lower', swap='')
    },
    skeleton_info={
        0: dict(link=('H1', 'T1'), color=[0, 255, 0]),
        1: dict(link=('T1', 'T2'), color=[0, 255, 0]),
        2: dict(link=('T2', 'T3'), color=[0, 255, 0]),
        3: dict(link=('T3', 'A2'), color=[0, 255, 0]),
        4: dict(link=('A2', 'A3'), color=[0, 255, 0]),
        5: dict(link=('A3', 'A4'), color=[0, 255, 0]),
        6: dict(link=('A4', 'A5'), color=[0, 255, 0]),
        7: dict(link=('A5', 'A6'), color=[0, 255, 0]),
        8: dict(link=('A6', 'A8'), color=[0, 255, 0]),
        9: dict(link=('A8', 'A9'), color=[0, 255, 0]),
    },
    joint_weights=[1.0] * 11,
    sigmas=[0.01] * 11,
    flip_pairs=[]
)