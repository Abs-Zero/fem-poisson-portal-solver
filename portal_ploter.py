import matplotlib.pyplot as plt
import numpy as np


def plot_portal(portals, fig_name):
    # fmt: off
    for portal_idx, portal in enumerate(portals):
        line1, line2, reverse = portal
        xm1, ym1, l1, theta1 = line1
        xm2, ym2, l2, theta2 = line2
        alpha1 = theta1 + np.pi / 2
        alpha2 = theta2 + np.pi / 2
        x11, y11, x12, y12 = (
            xm1 - l1 / 2 * np.cos(alpha1),
            ym1 - l1 / 2 * np.sin(alpha1),
            xm1 + l1 / 2 * np.cos(alpha1),
            ym1 + l1 / 2 * np.sin(alpha1),
        )
        sign = -1 if reverse else 1
        x21, y21, x22, y22 = (
            xm2 - sign * l2 / 2 * np.cos(alpha2),
            ym2 - sign * l2 / 2 * np.sin(alpha2),
            xm2 + sign * l2 / 2 * np.cos(alpha2),
            ym2 + sign * l2 / 2 * np.sin(alpha2),
        )

        plt.plot([x11, x12], [y11, y12], color="black", linewidth=1)
        plt.plot([x21, x22], [y21, y22], color="black", linewidth=1)
        
        l_argv = (l1 + l2) / 2
        
        arrow_len = l_argv / 8
        linewidth = arrow_len
        head_width = arrow_len / 6
        head_length = arrow_len / 6
        
        fontsize = l_argv * 4
        dx1, dy1 = arrow_len * np.cos(theta1), arrow_len * np.sin(theta1)
        dx2, dy2 = arrow_len * np.cos(theta2), arrow_len * np.sin(theta2)
        plt.arrow(xm1, ym1, dx1, dy1, head_width=head_width, head_length=head_length, 
                fc='black', ec='black', linewidth=linewidth)
        plt.arrow(xm2, ym2, dx2, dy2, head_width=head_width, head_length=head_length, 
                fc='black', ec='black', linewidth=linewidth)
        
        offset = 0.1
        plt.text(x11+offset, y11+offset, '0', fontsize=fontsize, color='black', ha='center')
        plt.text(x12+offset, y12+offset, '1', fontsize=fontsize, color='black', ha='center')
        plt.text(x21+offset, y21+offset, '0', fontsize=fontsize, color='black', ha='center')
        plt.text(x22+offset, y22+offset, '1', fontsize=fontsize, color='black', ha='center')
        plt.axis("equal")
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.savefig("./output/" + fig_name)
        plt.show()


p1 = [[(1, 0, 3, 0), (-1, 0, 3, np.pi / 6), False]]
p2 = [[(1, 0, 3, 0), (-1, 0, 3, np.pi / 6), True]]
plot_portal(p1, "传送门示意图，未翻转.svg")
plot_portal(p2, "传送门示意图，翻转.svg")