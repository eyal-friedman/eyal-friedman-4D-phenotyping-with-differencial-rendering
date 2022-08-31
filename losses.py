import torch
import torch.nn as nn
import numpy as np
import meshplex


# class IntersectionLoss(nn.Module):
#     def __init__(self, faces, faces_normals, average=False):
#         super(IntersectionLoss, self).__init__()
#
#         faces = faces.detach().cpu().numpy()
#         self.average = average
#
#         self.register_buffer('v0v', torch.from_numpy(faces[:, 0]).long())
#         self.register_buffer('v1v', torch.from_numpy(faces[:, 1]).long())
#         self.register_buffer('v2v', torch.from_numpy(faces[:, 2]).long())
#
#         # For testing - checking intersection with the first face only
#
#     def forward(self, vertices, eps=1e-6):
#         batch_size = vertices.size(0)
#         loss = []
#
#         if self.average:
#             return loss.sum() / batch_size
#         else:
#             return loss
def intersect(vp0, vp1, vp2, vd0, vd1, vd2, eps):
    # intersect0 = VV0+(VV1-VV0)*D0/(D0-D1+eps);
    tmp0 = vd0 / (vd0 - vd1 + eps)
    intersect0 = vp0 + (vp1 - vp0) * tmp0

    # intersect1 = VV0+(VV2-VV0)*D0/(D0-D2+eps);
    tmp1 = vd0 / (vd0 - vd2 + eps)
    intersect1 = vp0 + (vp2 - vp0) * tmp1
    return intersect0, intersect1

class VolumeLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(VolumeLoss, self).__init__()

        faces = faces.detach().cpu().numpy()
        #self.nf = faces.size(0)
        self.average = average
        #self.faces = faces


        self.register_buffer('v0v', torch.from_numpy(faces[:, 0]).long())
        self.register_buffer('v1v', torch.from_numpy(faces[:, 1]).long())
        self.register_buffer('v2v', torch.from_numpy(faces[:, 2]).long())

        #self.register_buffer('facesV', self.faces)

    def forward(self, vertices, faces_normals, eps=1e-4):
        batch_size = vertices.size(0)

        loss_list = []
        loss = []
        # -------------------------------------------------
        ## Original volume loss - unmark to apply
        #loss = torch.sum(torch.sum(vertices[:,self.v0v,:] * torch.cross(vertices[:, self.v1v, :], vertices[:, self.v2v, :]), dim=2)/6)
        # -------------------------------------------------

        boolFullTable = torch.zeros((self.v0v.shape[0], self.v0v.shape[0]), dtype=bool).cuda()
        DU_FullTable = torch.zeros((self.v0v.shape[0], self.v0v.shape[0], 3), dtype=torch.float32).cuda()
        Up_FullTable = torch.zeros((self.v0v.shape[0], self.v0v.shape[0], 3), dtype=torch.float32).cuda()


        ## Second try - NoDivTriTriIntersection

        for faceNumber in range(self.v0v.shape[0]):

            p0 = vertices[:, self.v0v[faceNumber], :]
            # Using face_normals instead of V1V0 * V2V0 cross product
            # compute plane equation of triangle i_th(faceNumber)
            # plane equation 1: N1.X+d1=0 */
            n1 = faces_normals[:, faceNumber]
            #d1 = -torch.dot(p0, n1)
            d1 = -torch.mm(p0, n1.T)

            U0 = vertices[:, self.v0v, :]
            U1 = vertices[:, self.v1v, :]
            U2 = vertices[:, self.v2v, :]

            # du0 = torch.dot(U0, n1)+d1
            dU0 = torch.sum(U0 * n1, dim=2) + d1
            dU1 = torch.sum(U1 * n1, dim=2) + d1
            dU2 = torch.sum(U2 * n1, dim=2) + d1

            DU_FullTable[faceNumber, :, 0] = dU0
            DU_FullTable[faceNumber, :, 1] = dU1
            DU_FullTable[faceNumber, :, 2] = dU2

            # dU0_test = torch.matmul(U0, n1.T).resize(1, U0.shape[1])
            # testing matmul instead of 8

            distance = torch.cat((dU0.T, dU1.T, dU2.T), dim=1)

            # false to all the trivial rejections, true for further checking
            allBool = torch.bitwise_not(torch.bitwise_or(torch.all(distance + eps > 0, dim=1, keepdim=True),
                                       (torch.all(distance - eps < 0, dim=1, keepdim=True))))

            boolFullTable[faceNumber] = allBool.T

            #######################################
            # -> a vectorized form

            # Computing intersection line
            D = torch.cross(n1.repeat(1, faces_normals.shape[1], 1), faces_normals)
            D = torch.nn.functional.normalize(D, dim=2)

            # Finding pv the projection of the points on L
            # proejecting both triangles on L
            U0p = torch.sum(D * U0, dim=2)
            U1p = torch.sum(D * U1, dim=2)
            U2p = torch.sum(D * U2, dim=2)

            Up_FullTable[faceNumber, :, 0] = U0p
            Up_FullTable[faceNumber, :, 1] = U1p
            Up_FullTable[faceNumber, :, 2] = U2p

            # Our relevant triangle projection is the faceNumber index
            # dUi is the distance from all the triangles' vertices to our triangle plane.
            # t1 = U0p + (U1p-U0p)(dU0/(dU0-dU1))
            # t2 = U1p + (U2p-U1p)(dU1/(dU1-dU2))
            # t1-t2 is the intersection interval
            # for i, up0123 in enumerate(zip(U0p[0], U1p[0], U2p[0])):
            #     if i == faceNumber:
            #         continue
            #     if allBool[i]:
            #         print(i)
            # we need to find the largetst component of D
            #a, b = torch.max(d, 2, keepdim=True)
            #torch.index_select(d, 2, b[0])

        ## applying torch and to BoolFullTable and BoolFullTable.T to make sure when both triangle's planes intersect.
        boolFullTable = torch.bitwise_and(boolFullTable, boolFullTable.T)

        for i in range(boolFullTable.shape[0]):
            for j in range(boolFullTable.shape[1]):
                if i < j:
                    # Skipping the rejected triangles in the fast rejection test.
                    if not boolFullTable[i, j]:
                        continue

                    #print(i, j)
                    # time to check intervals of [i,j] and [j,i] triangles on L[i;j]

                    # vp0 = Up_FullTable[i, j, 0]
                    # vp1 = Up_FullTable[i, j, 1]
                    # vp2 = Up_FullTable[i, j, 2]

                    vp0, vp1, vp2 = Up_FullTable[i, j, :]
                    vd0, vd1, vd2 = DU_FullTable[i, j, :]
                    interval0 = []
                    # interval0 = intersect(vp0, vp1, vp2, vd0, vd1, vd2, eps)

                    if (vd0 * vd1) > 0:
                        # here we know that D0D2 <= 0 that meas D0, D1 are on the same side of the line
                        # intersect0 = VV2+(VV0-VV2)*D2/(D2-D0+eps);
                        # intersect1 = VV2+(VV1-VV2)*D2/(D2-D1+eps);
                        interval0 = intersect(vp2, vp0, vp1, vd2, vd0, vd1, eps)

                        # tmp0 = DU_FullTable[i, j, 2] / (DU_FullTable[i, j, 2] - DU_FullTable[i, j, 0] + eps)
                        # intersect0 = Up_FullTable[i, j, 2] + (Up_FullTable[i, j, 0]-Up_FullTable[i, j, 2]) * tmp0
                        # tmp1 = DU_FullTable[i, j, 2] / (DU_FullTable[i, j, 2] - DU_FullTable[i, j, 1] + eps)
                        # intersect1 = Up_FullTable[i, j, 2] + (Up_FullTable[i, j, 1] - Up_FullTable[i, j, 2]) * tmp1
                        # print(i, j, 'Case 1, interval0:')
                        # print(interval0)
                        # print(i, j, 'Case 1, intersect check:')
                        # print(intersect0, intersect1)

                    elif (vd0 * vd2) > 0:
                        # here we know that D0D1 <= 0 that meas D0, D2 are on the same side of the line
                        # intersect0 = VV1+(VV0-VV1)*D1/(D1-D0+eps);
                        # intersect1 = VV1+(VV2-VV1)*D1/(D1-D2+eps);
                        interval0 = intersect(vp1, vp0, vp2, vd1, vd0, vd2, eps)

                        # tmp0 = DU_FullTable[i, j, 1] / (DU_FullTable[i, j, 1] - DU_FullTable[i, j, 0] + eps)
                        # intersect0 = Up_FullTable[i, j, 1] + (Up_FullTable[i, j, 0] - Up_FullTable[i, j, 1]) * tmp0
                        # tmp1 = DU_FullTable[i, j, 1] / (DU_FullTable[i, j, 1] - DU_FullTable[i, j, 2] + eps)
                        # intersect1 = Up_FullTable[i, j, 1] + (Up_FullTable[i, j, 2] - Up_FullTable[i, j, 1]) * tmp1
                        # print(i, j, 'Case 2, interval0:')
                        # print(interval0)
                        # print(i, j, 'Case 2, intersect check:')
                        # print(intersect0, intersect1)

                    else: #(DU_FullTable[i, j, 1] * DU_FullTable[i, j, 2]) > 0:
                        # here we know that D0D1 <= 0 that meas D1, D2 are on the same side of the line
                        # intersect0 = VV0+(VV1-VV0)*D0/(D0-D1+eps);
                        # intersect1 = VV0+(VV2-VV0)*D0/(D0-D2+eps);
                        interval0 = intersect(vp0, vp1, vp2, vd0, vd1, vd2, eps)

                        # tmp0 = DU_FullTable[i, j, 0] / (DU_FullTable[i, j, 0] - DU_FullTable[i, j, 1] + eps)
                        # intersect0 = Up_FullTable[i, j, 0] + (Up_FullTable[i, j, 1] - Up_FullTable[i, j, 0]) * tmp0
                        # tmp1 = DU_FullTable[i, j, 0] / (DU_FullTable[i, j, 0] - DU_FullTable[i, j, 2] + eps)
                        # intersect1 = Up_FullTable[i, j, 0] + (Up_FullTable[i, j, 2] - Up_FullTable[i, j, 0]) * tmp1
                        # print(i, j, 'Case 3, interval0:')
                        # print(interval0)
                        # print(i, j, 'Case 3, intersect check:')
                        # print(intersect0, intersect1)

                    up0, up1, up2 = -Up_FullTable[j, i, :]
                    ud0, ud1, ud2 = DU_FullTable[j, i, :]
                    interval1 = []
                    if (ud0 * ud1) > 0:
                        # here we know that D0D2 <= 0 that meas D0, D1 are on the same side of the line
                        # intersect0 = VV2+(VV0-VV2)*D2/(D2-D0+eps);
                        # intersect1 = VV2+(VV1-VV2)*D2/(D2-D1+eps);
                        interval1 = intersect(up2, up0, up1, ud2, ud0, ud1, eps)
                        # print(j, i, 'Case 1, interval1:')
                        # print(interval1)

                    elif (ud0 * ud2) > 0:
                        # here we know that D0D1 <= 0 that meas D0, D2 are on the same side of the line
                        # intersect0 = VV1+(VV0-VV1)*D1/(D1-D0+eps);
                        # intersect1 = VV1+(VV2-VV1)*D1/(D1-D2+eps);
                        interval1 = intersect(up1, up0, up2, ud1, ud0, ud2, eps)
                        # print(j, i, 'Case 2, interval1:')
                        # print(interval1)

                    else: #(DU_FullTable[i, j, 1] * DU_FullTable[i, j, 2]) > 0:
                        # here we know that D0D1 <= 0 that meas D1, D2 are on the same side of the line
                        # intersect0 = VV0+(VV1-VV0)*D0/(D0-D1+eps);
                        # intersect1 = VV0+(VV2-VV0)*D0/(D0-D2+eps);
                        interval1 = intersect(up0, up1, up2, ud0, ud1, ud2, eps)

                        # print(j, i, 'Case 3, interval1:')
                        # print(interval1)

                    if torch.bitwise_or(max(interval0) <= min(interval1) + 0.01,
                                    (max(interval1) <= min(interval0) + 0.01)):
                        #print('not intersection at: %i %i' % (i, j))
                        boolFullTable[i, j] = False
                        boolFullTable[j, i] = False
                    else:
                        #print('!!! Intersection at: %i %i !!!' % (i, j))
                        pass





        # ## looping through all the faces in the mesh - first try!
        # for t0 in range(self.v0v.shape[0]):
        #     P0 = vertices[:, self.v0v[t0], :]
        #     PQ0 = vertices[:, self.v0v, :] - P0
        #     PQ1 = vertices[:, self.v1v, :] - P0
        #     PQ2 = vertices[:, self.v2v, :] - P0
        #
        #     dis0 = torch.sum(PQ0 * faces_normals[:, t0], dim=2)
        #     dis1 = torch.sum(PQ1 * faces_normals[:, t0], dim=2)
        #     dis2 = torch.sum(PQ2 * faces_normals[:, t0], dim=2)
        #
        #     distance = torch.cat((dis0.T, dis1.T, dis2.T), dim=1)
        #
        #     # allBool = torch.all(distance>0, dim=2, keepdim=True)
        #     allBool = torch.bitwise_or(torch.all(distance + eps > 0, dim=1, keepdim=True),
        #                                (torch.all(distance - eps < 0, dim=1, keepdim=True)))
        #     allBoolFull = allBool.repeat(1, 1, 3)
        #     trivial_rejection = torch.bitwise_not(
        #         allBoolFull) * distance  # -> this is the mask after rejecting the trivial cases.
        #     loss_list.append(torch.sum(trivial_rejection))

        # # check the first triangle with all the rest - P is vertex0 of the first face.
        # P0 = vertices[:, self.v0v[0], :]
        # P1 = vertices[:, self.v1v[0], :]
        # P2 = vertices[:, self.v2v[0], :]
        #
        # # Q0 = vertices[:, self.v0v, :] - the t0 vertices of each face an so on
        # PQ0 = vertices[:, self.v0v, :] - P0
        # PQ1 = vertices[:, self.v1v, :] - P0
        # PQ2 = vertices[:, self.v2v, :] - P0
        #
        # dis0 = torch.sum(PQ0 * faces_normals[:, 0], dim=2)
        # dis1 = torch.sum(PQ1 * faces_normals[:, 0], dim=2)
        # dis2 = torch.sum(PQ2 * faces_normals[:, 0], dim=2)
        #
        # distance = torch.cat((dis0.T, dis1.T, dis2.T), dim=1)
        #
        # #allBool = torch.all(distance>0, dim=2, keepdim=True)
        # allBool = torch.bitwise_or(torch.all(distance + eps > 0, dim=1, keepdim=True),
        #                            (torch.all(distance - eps < 0, dim=1, keepdim=True)))
        # allBoolFull = allBool.repeat(1, 1, 3)
        # trivial_rejection = torch.bitwise_not(allBoolFull) * distance # -> this is the mask after rejecting the trivial cases.
        # loss = torch.sum(trivial_rejection)

        ## torch.dot(PQ[0,6],faces_normals[0,6]) -> that works for a single check


        ###test####
        # vertices[:, self.v0v, :]
        # torch.dot(self.v0v, torch.cross(vertices[:, self.v1v, :], vertices[:, self.v2v, :]))

        loss = torch.sum(vertices[:, self.v0v, 0] * torch.sum(boolFullTable, dim=1)) + \
            torch.sum(vertices[:, self.v2v, 0] * torch.sum(boolFullTable, dim=1))  + \
               torch.sum(vertices[:, self.v2v, 0] * torch.sum(boolFullTable, dim=1))
        loss = loss / 3
        #loss = torch.stack(loss_list).sum()


        #loss = loss.sum()
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        #print(x)
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x


class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average

        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss
