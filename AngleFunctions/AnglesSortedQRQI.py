from Rodrigues import *

def AnglesSortedQRQI(QRstore,QIstore, Frequencies):
    N = len(Frequencies)
    SortedAnglesstore=np.zeros(N)
    for n in range(N):
        QR = np.zeros((3,3))
        QI = np.zeros((3,3))
        uR =np.zeros(3)
        for i in range(3):
            for j in range(3):
                QR[i,j] = QRstore[n,i,j]
                QI[i,j] = QIstore[n,i,j]

        # In this case input argument Q and I are already sorted as desired and so only compute Angles
        angle, K, tvec = Rodrigues(QR,QI)
        SortedAnglesstore[n] = angle
    return SortedAnglesstore
