# Usage example: python3 vis_grasp.py student_grasps_v1/02808440/148ec8030e52671d44221bef0fa3c36b/0/
from pathlib import Path
# import pybullet
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("data_path")
args = parser.parse_args()

data = np.load(Path(args.data_path) / "recording.npz")

# Inspect each array
for key in data.files:
    print(f"Array '{key}': shape {data[key].shape}, dtype {data[key].dtype}")
    # print(data[key][:10])
    # exit(1)


exit(1)

pybullet.connect(pybullet.GUI)

# Load hand
hand_id = pybullet.loadURDF(
    "urdfs/dlr2.urdf",
    globalScaling=1,
    basePosition=[0, 0, 0],
    baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
)

# Load object
visualShapeId = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=str(Path(args.data_path) / "mesh.obj"),
                rgbaColor=[1,1,1,1],
                specularColor=[0.4, .4, 0],
                visualFramePosition=[0, 0, 0],
                meshScale=1)
object_id = pybullet.createMultiBody(baseMass=1,
                    baseInertialFramePosition=[0, 0, 0],
                    baseVisualShapeIndex=visualShapeId,
                    baseCollisionShapeIndex=visualShapeId,
                    basePosition=[0,0,0],
                    baseOrientation=[0,0,0,1])
                        
# Load grasps

# Sort by grasp score
sorted_indx = np.argsort(data["scores"])[::-1]
print(data["grasps"].shape)

for i in sorted_indx:
    grasp = data["grasps"][i]

    # Set hand pose
    pybullet.resetBasePositionAndOrientation(bodyUniqueId=hand_id, posObj=grasp[:3], ornObj=grasp[3:7])
    
    # Set joint angles
    for k, j in enumerate([1,2,3, 7,8,9, 13,14,15, 19,20,21]):
        pybullet.resetJointState(hand_id, jointIndex=j, targetValue=grasp[7 + k], targetVelocity=0)
        # Set coupled joint
        if j in [3, 9, 15, 21]:
            pybullet.resetJointState(hand_id, jointIndex=j + 1, targetValue=grasp[7 + k], targetVelocity=0)
    
    print(f"Score {data['scores'][i]}")
    input()
