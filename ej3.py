from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({  # Permite usar LaTeX en los plots
    "text.usetex": True,
    "font.family": "Helvetica"
})


def polar2z(r, theta):
    return r * np.exp(1j * theta)


def z2polar(z):
    return np.abs(z), np.angle(z)


halignmap = {
    'c': 'center',
    'r': 'right',
    'l': 'left'
}
valignmap = {
    'c': 'center',
    't': 'top',
    'b': 'bottom'
}


def plot_referenceFrame_fromWorld(pose, axs=None, symbol="", align="cc", refFrame_World=(0, 0, 0)):
    if axs is None:
        axs = plt

    rang = np.deg2rad(refFrame_World[2]);
    rfX = np.exp(1j * rang)  # ref frame X axis
    rfY = rfX * 1j  # ref frame Y axis
    rotPoint = (rfX * pose[0] + rfY * pose[1])

    ang = np.deg2rad(pose[2]) + rang;
    fX = np.exp(1j * ang)  # frame X axis
    fY = fX * 1j  # frame Y axis
    axs.quiver((refFrame_World[0], refFrame_World[0] + np.real(rotPoint), refFrame_World[0] + np.real(rotPoint)),
               (refFrame_World[1], refFrame_World[1] + np.imag(rotPoint), refFrame_World[1] + np.imag(rotPoint)),
               (np.real(rotPoint), np.real(fX), np.real(fY)),
               (np.imag(rotPoint), np.imag(fX), np.imag(fY)),
               color=['k', 'r', 'g'],
               angles='xy', scale_units='xy', scale=1)
    axs.text(refFrame_World[0] + np.real(rotPoint), refFrame_World[1] + np.imag(rotPoint), "$" + symbol + "$",
             horizontalalignment=halignmap[align[0]], verticalalignment=valignmap[align[1]], color='k')
    axs.text(refFrame_World[0] + np.real(rotPoint) + np.real(fX), refFrame_World[1] + np.imag(rotPoint) + np.imag(fX),
             r"$x_" + symbol + "$",
             horizontalalignment='left', verticalalignment='top', color='r')
    axs.text(refFrame_World[0] + np.real(rotPoint) + np.real(fY), refFrame_World[1] + np.imag(rotPoint) + np.imag(fY),
             r"$y_" + symbol + "$",
             horizontalalignment='right', verticalalignment='top', color='g')


def plot_point(position, axs=None, symbol="", align="cc", refFrame_World=(0, 0, 0)):
    if axs is None:
        axs = plt

    ang = np.deg2rad(refFrame_World[2]);
    fX = np.exp(1j * ang)  # frame X axis
    fY = fX * 1j  # frame Y axis
    rotPoint = (fX * position[0] + fY * position[1])

    axs.quiver(refFrame_World[0], refFrame_World[1], np.real(rotPoint), np.imag(rotPoint),
               color='k',
               angles='xy', scale_units='xy', scale=1)
    axs.text(refFrame_World[0] + np.real(rotPoint), refFrame_World[1] + np.imag(rotPoint), symbol,
             horizontalalignment=halignmap[align[0]], verticalalignment=valignmap[align[1]],
             color='k')


if __name__ == "__main__":
    poseA_W = np.array((2, 3, 45))
    poseB_A = np.array((1, 1, -45))
    p1_W = np.array((1, 5))
    p2_A = np.array((1, 2))

    thA = np.deg2rad(poseA_W[2])
    xi_A_W = np.array(((np.cos(thA), -np.sin(thA), poseA_W[0]),
                       (np.sin(thA), np.cos(thA), poseA_W[1]),
                       (0, 0, 1)))
    thB = np.deg2rad(poseB_A[2])
    xi_B_A = np.array(((np.cos(thB), -np.sin(thB), poseB_A[0]),
                       (np.sin(thB), np.cos(thB), poseB_A[1]),
                       (0, 0, 1)))

    xi_W_A = np.linalg.inv(xi_A_W)
    xi_A_B = np.linalg.inv(xi_B_A)

    # print(xi_A_W @ xi_B_A @ np.array((0, 0, 1)))

    ax = plt.axes()
    plt.xlim(-2, 5)
    plt.xlabel(r"$x$")
    plt.ylim(-1, 6)
    plt.ylabel(r"$y$")
    ax.set_aspect('equal', 'box')
    plt.grid(visible=True, alpha=0.5)

    plot_referenceFrame_fromWorld((0, 0, 0), ax, symbol="W", align='rt')  # Plot World
    plot_referenceFrame_fromWorld(poseA_W, ax, symbol="A", align='lt')
    plot_referenceFrame_fromWorld(poseB_A, ax, symbol="B", align='lt', refFrame_World=poseA_W)
    # plot_referenceFrame_fromWorld(poseB_W, ax, symbol="B", align='rb')
    plot_point(p1_W, ax, symbol=r"${}^Wp1$", align='rb')
    # plot_point(p2_W, ax, symbol=r"${}^Wp2$", align='rb')
    plot_point(p2_A, ax, symbol=r"${}^Ap2$", align='lc', refFrame_World=poseA_W)
    # plot_point((1, 1), ax, symbol=r"$B$", align='lc', refFrame_World=poseA_W)

    f = plt.gcf()
    # f = ""
    # plt.show(block=False)
    if True:
        print("Mostrar Figura? s/[N]: ")
        if input() == "s":
            plt.show(block=True)
            plt.figure(f)
        print("Guardar Figura? s/[N]: ")
        if input() == "s":
            plt.savefig("ej3a.png")

    print("Item b _____________________________")
    p2_W = (xi_A_W @ np.append(p2_A, 1))[0:2]
    p1_A = (xi_W_A @ np.append(p1_W, 1))[0:2]

    p2_B = (xi_W_B @ np.append(p2_W, 1))[0:2]

    print("p1_W=" + str(p1_W) + " --> " + "p1_A=" + str(p1_A))
    print("p2_A=" + str(p2_A) + " --> " + "p2_W=" + str(p2_W) + " --> " + "p2_B=" + str(p2_B))
