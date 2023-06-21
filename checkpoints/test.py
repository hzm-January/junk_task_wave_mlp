import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

import numpy as np
visibility = [1,0,1]
w_=np.array([[1,2,3],[4,5,6],[7,8,9]]).T
print(w_)
w=w_[np.array(visibility)==1].T
print(w)

cam_w_extrinsics = [[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [0, 0, 0, 1]]
print(cam_w_extrinsics[-1][-1])
torch.tensor(cam_w_extrinsics).float().cuda()

# maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
#                                             [-1, 0, 0, 0],
#                                             [0, -1, 0, 0],
#                                             [0, 0, 0, 1]], dtype=float)
# cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w
# print(cam_extrinsics)
#
# A = torch.tensor([[1,2,1,2],[3,4,3,4],[5,6,5,6],[7,8,7,8],[9,10,9,10]])
# print(A.shape)
# print(A)
# B = A.view((5,2,2))
# print(B.shape)
# print(B)