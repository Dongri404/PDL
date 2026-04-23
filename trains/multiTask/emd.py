import torch

import torch.nn as nn



# Adapted from https://github.com/gpeyre/SinkhornAutoDiff

class SinkhornDistance(nn.Module):

    r"""

    Given two empirical measures each with :math:`P_1` locations

    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,

    outputs an approximation of the regularized OT cost for point clouds.

    Args:

        eps (float): regularization coefficient

        max_iter (int): maximum number of Sinkhorn iterations

        reduction (string, optional): Specifies the reduction to apply to the output:

            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,

            'mean': the sum of the output will be divided by the number of

            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:

        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`

        - Output: :math:`(N)` or :math:`()`, depending on `reduction`

    """

    def __init__(self, eps, max_iter, reduction='none'):

        super(SinkhornDistance, self).__init__()

        self.eps = eps

        self.max_iter = max_iter

        self.reduction = reduction



    def forward(self, x, y):

        # The Sinkhorn algorithm takes as input three variables :

        C = self._cost_matrix(x, y)  # Wasserstein cost function

        # # Cx = self._cost_matrix(x, x)
        # # Cy = self._cost_matrix(y, y)
        # Cx = (1 - torch.matmul(x, torch.t(x))) / 2
        # Cy = (1 - torch.matmul(y, torch.t(y))) / 2

        # one_mat = torch.zeros(C.size(0), C.size(1)).cuda()

        # num_pos = 4

        # for i in range(int(C.size(0)/num_pos)):

        # 	one_mat[i:num_pos*(i+1), i:num_pos*(i+1)] = 1.0

        # C = C * one_mat



        x_points = x.shape[-2]  #倒数第二个维度48

        y_points = y.shape[-2]

        if x.dim() == 2:

            batch_size = 1

        else:

            batch_size = x.shape[0]



        # both marginals are fixed with equal weights

        mu = torch.empty(batch_size, x_points, dtype=torch.float,   #mv

                         requires_grad=False).fill_(1.0 / x_points).squeeze()

        nu = torch.empty(batch_size, y_points, dtype=torch.float,   #mt

                         requires_grad=False).fill_(1.0 / y_points).squeeze()



        mu = mu.cuda()

        nu = nu.cuda()



        u = torch.zeros_like(mu).cuda() #v

        v = torch.zeros_like(nu).cuda() #t

        # To check if algorithm terminates because of threshold

        # or max iterations reached

        actual_nits = 0

        # Stopping criterion

        thresh = 1e-1



        # Sinkhorn iterations

        for i in range(self.max_iter):

            u1 = u  # useful to check the update

            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v

            err = (u - u1).abs().sum(-1).mean()



            actual_nits += 1

            if err.item() < thresh:

                break



        U, V = u, v

        # Transport plan pi = diag(a)*K*diag(b)

        pi = torch.exp(self.M(C, U, V))

        ### Sinkhorn distance

        # # #################只留同类的相似性###############
        #
        # one_mat = torch.zeros(C.size(0), C.size(1)).cuda()
        #
        # num_pos = 8
        #
        # for i in range(int(C.size(0)/num_pos)):
        #
        # 	one_mat[i:num_pos*(i+1), i:num_pos*(i+1)] = 1.0
        #
        # Cx = Cx * one_mat
        # Cy = Cy * one_mat
        # # #################只留同类的相似性###############
        # costxy = torch.sum(pi * (Cx+Cy), dim=(-2, -1))

        cost = torch.sum(pi * C, dim=(-2, -1))


        ###############################直接根据相似性求权重##############################
        # # S_mat = torch.matmul(x, torch.t(y))
        # # S_mat = (C.max() - C)
        # S_mat = (1 - C)
        # # S_mat = C
        #
        # # pi = torch.nn.functional.softmax(S_mat, dim=1) / 48 #SYSU
        # pi = torch.nn.functional.softmax(S_mat, dim=1) / 24  # REG
        # cost = torch.sum(pi * C, dim=(-2, -1))
        ###############################直接根据相似性求权重##############################


        if self.reduction == 'mean':

            cost = cost.mean()

        elif self.reduction == 'sum':

            cost = cost.sum()



        return cost, pi, C
        # return cost + 0.1*costxy, pi, C



    def M(self, C, u, v):

        "Modified cost for logarithmic updates"

        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps



    @staticmethod

    def _cost_matrix(x, y, p=2):

        "Returns the matrix of $|x_i-y_j|^p$."

        x_col = x.unsqueeze(-2)

        y_lin = y.unsqueeze(-3)

        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

        return C



    @staticmethod

    def ave(u, u1, tau):

        "Barycenter subroutine, used by kinetic acceleration through extrapolation."

        return tau * u + (1 - tau) * u1