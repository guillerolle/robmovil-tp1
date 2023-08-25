from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({  # Permite usar LaTeX en los plots
    "text.usetex": True,
    "font.family": "Helvetica"
})

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


class Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0, referenceFrame=None, symbol=""):
        self.x = x
        self.y = y
        self.theta = theta
        self.frame = referenceFrame
        self.symbol = symbol

    def getPoseV(self):
        return np.ndarray((self.x, self.y, self.theta))

    def getPoseZ(self):
        return self.x + 1j * self.y

    def getPosition(self):
        return np.array((self.x, self.y))

    def getPositionInhomogeneous(self):
        return np.array((self.x, self.y, 1))

    def drawPoseVector(self, axs=plt, color='k'):
        frame_world_coords = Pose(x=0, y=0, theta=0)
        if self.frame is not None:
            frame_world_coords = self.frame.pose.getWorldCoords()
        self_world_coords = self.getWorldCoords()
        axs.quiver(frame_world_coords.x, frame_world_coords.y,
                   self_world_coords.x - frame_world_coords.x, self_world_coords.y - frame_world_coords.y,
                   color=color,
                   angles='xy', scale_units='xy', scale=1)
        axs.text(self_world_coords.x, self_world_coords.y, self.symbol,
                 horizontalalignment='right', verticalalignment='top', color='k')
        # plt.show()

    def getWorldCoords(self) -> "Pose":
        if self.frame is not None:
            vector = self.frame.homogeneous_fromme_toworld() @ self.getPositionInhomogeneous()
            angle = self.frame.angle_fromworld() + self.theta
            return Pose(x=vector[0], y=vector[1], theta=angle)
        else:
            return self

    def __str__(self):
        return "x= " + str(self.x) + "\ty= " + str(self.y) + "\tdeg= " + str(np.rad2deg(self.theta))


class ReferenceFrame:
    def __init__(self, x=0.0, y=0.0, theta=0.0, parent: "ReferenceFrame" = None, symbol="", align="cc"):
        self.pose = Pose(x, y, theta, parent)
        self.parent = parent
        self.symbol = symbol
        self.align = align

    def homogeneous_fromme_toparent(self):
        return np.array(((np.cos(self.pose.theta), -np.sin(self.pose.theta), self.pose.x),
                         (np.sin(self.pose.theta), np.cos(self.pose.theta), self.pose.y),
                         (0, 0, 1)))

    def homogeneous_fromparent_tome(self):
        return np.linalg.inv(self.homogeneous_fromme_toparent())

    def drawReferenceFrame(self, axs=plt):
        origin = self.pose.getWorldCoords()
        xAxis = Pose(x=1, y=0, referenceFrame=self)
        yAxis = Pose(x=0, y=1, referenceFrame=self)
        # origin.drawPoseVector()
        xAxis.drawPoseVector(color='r')
        yAxis.drawPoseVector(color='g')
        self.pose.drawPoseVector(color='b')
        axs.text(origin.x, origin.y, self.symbol,
                 horizontalalignment=halignmap[self.align[0]], verticalalignment=valignmap[self.align[1]], color='k')
        axs.text(xAxis.getWorldCoords().x, xAxis.getWorldCoords().y, r"$x_{" + self.symbol[1:-1] + "}$",
                 horizontalalignment='left', verticalalignment='top', color='r')
        axs.text(yAxis.getWorldCoords().x, yAxis.getWorldCoords().y, r"$y_{" + self.symbol[1:-1] + "}$",
                 horizontalalignment='right', verticalalignment='top', color='g')

    def homogeneous_fromme_toworld(self):
        # Calcula transformacion homogenea del Sistema=Self a World=None
        frame2world = np.identity(3)
        curFrame: ReferenceFrame = self
        while curFrame is not None:
            # Premultiplica transformaciones homogeneas
            frame2world = curFrame.homogeneous_fromme_toparent() @ frame2world
            curFrame = curFrame.parent
        return frame2world

    def homogeneous_fromworld_tome(self):
        return np.linalg.inv(self.homogeneous_fromme_toworld())

    def angle_fromworld(self):
        angle = 0
        curFrame: ReferenceFrame = self
        while curFrame is not None:
            angle += curFrame.pose.theta
            curFrame = curFrame.parent
        return angle


def ej3a(showFig=False, saveFig=False):
    refWorld = ReferenceFrame(x=0, y=0, theta=0, symbol=r"$W$", align='rt')
    refA = ReferenceFrame(x=2, y=3, theta=np.deg2rad(45), parent=refWorld, symbol=r"$A$", align='lt')
    refB = ReferenceFrame(x=1, y=1, theta=np.deg2rad(-45), parent=refA, symbol=r"$B$", align='lt')

    p1 = Pose(x=1, y=5, referenceFrame=refWorld, symbol=r"$P1$")
    p2 = Pose(x=1, y=2, referenceFrame=refA, symbol=r"$P2$")
    # p3 = Pose(x=1, y=1, referenceFrame=refB, symbol=r"$P3$")

    ax = plt.axes()
    plt.xlim(-2, 5)
    plt.xlabel(r"$x$")
    plt.ylim(-1, 6)
    plt.ylabel(r"$y$")
    ax.set_aspect('equal', 'box')
    plt.grid(visible=True, alpha=0.5)

    p1.drawPoseVector()
    p2.drawPoseVector()
    # p3.drawPoseVector()

    refWorld.drawReferenceFrame()
    refA.drawReferenceFrame()
    refB.drawReferenceFrame()

    f = plt.gcf()
    if showFig:
        plt.show(block=True)
        plt.figure(f)
    if saveFig:
        plt.savefig("ej3a.png", dpi=400)

    print("Item b _____________________________")
    xi_fromW_toA = refA.homogeneous_fromworld_tome()
    xi_fromA_toB = refB.homogeneous_fromparent_tome()

    p1_A = (xi_fromW_toA @ p1.getPositionInhomogeneous())[0:2]
    p2_B = (xi_fromA_toB @ p2.getPositionInhomogeneous())[0:2]

    print("p1_A=" + str(p1_A))
    print("p2_B=" + str(p2_B))
    print("poseB_fromW:\t" + str(refB.pose.getWorldCoords()))


if __name__ == "__main__":
    ej3a(showFig=True, saveFig=False)
