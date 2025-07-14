# --- dataset_info 定義 -----------------------
dataset_info = dict(
    dataset_name='silkworm11',
    paper_info=dict(title='Silkworm Pose', year=2025),
    keypoint_info={
        0: dict(name='head',  id=0,  color=[255,0,0], type='upper',  swap=''),
        1: dict(name='T1_mid',id=1,  color=[255,0,0], type='upper',  swap=''),
        2: dict(name='T2_mid',id=2,  color=[255,0,0], type='upper',  swap=''),
        3: dict(name='T3_mid',id=3,  color=[255,0,0], type='upper',  swap=''),
        4: dict(name='A2_mid',id=4,  color=[255,0,0], type='lower',  swap=''),
        5: dict(name='A3_mid',id=5,  color=[255,0,0], type='lower',  swap=''),
        6: dict(name='A4_mid',id=6,  color=[255,0,0], type='lower',  swap=''),
        7: dict(name='A5_mid',id=7,  color=[255,0,0], type='lower',  swap=''),
        8: dict(name='A6_mid',id=8,  color=[255,0,0], type='lower',  swap=''),
        9: dict(name='A8_mid',id=9,  color=[255,0,0], type='lower',  swap=''),
       10: dict(name='tail',  id=10, color=[255,0,0], type='lower',  swap='')
    },
    skeleton_info={
        0: dict(link=('head','T1_mid'),    color=[0,255,0]),
        1: dict(link=('T1_mid','T2_mid'),  color=[0,255,0]),
        2: dict(link=('T2_mid','T3_mid'),  color=[0,255,0]),
        3: dict(link=('T3_mid','A2_mid'),  color=[0,255,0]),
        4: dict(link=('A2_mid','A3_mid'),  color=[0,255,0]),
        5: dict(link=('A3_mid','A4_mid'),  color=[0,255,0]),
        6: dict(link=('A4_mid','A5_mid'),  color=[0,255,0]),
        7: dict(link=('A5_mid','A6_mid'),  color=[0,255,0]),
        8: dict(link=('A6_mid','A8_mid'),  color=[0,255,0]),
        9: dict(link=('A8_mid','tail'),    color=[0,255,0])
    },
    joint_weights=[1.0]*11,
    sigmas=[0.01]*11,
    flip_pairs=[]
)
# --------------------------------------------