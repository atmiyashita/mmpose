dataset_info = dict(
    dataset_name='silkworm4',
    paper_info=dict(
        title='Silkworm 4-Point Pose Estimation Dataset',
        homepage='',
        year='2025',
    ),
    keypoint_info={
        0: dict(name='H1', id=0, color=[255, 0, 0], type='upper', swap=''),      # Head
        1: dict(name='T1', id=1, color=[255, 0, 0], type='upper', swap=''),      # Thorax start  
        2: dict(name='A2', id=2, color=[255, 0, 0], type='lower', swap=''),      # Abdomen start
        3: dict(name='A9', id=3, color=[255, 0, 0], type='lower', swap=''),      # Tail
    },
    skeleton_info={
        0: dict(link=('H1', 'T1'), id=0, color=[0, 255, 0]),     # Head to thorax
        1: dict(link=('T1', 'A2'), id=1, color=[0, 255, 0]),     # Thorax to abdomen
        2: dict(link=('A2', 'A9'), id=2, color=[0, 255, 0]),     # Abdomen to tail
    },
    joint_weights=[1.0, 1.0, 1.0, 1.0],  # 4 keypoints
    sigmas=[0.01, 0.01, 0.01, 0.01],     # 4 keypoints
    flip_pairs=[],  # No flip pairs for silkworm
)
