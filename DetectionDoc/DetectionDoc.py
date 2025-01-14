import cv2
import numpy as np

def scanner_document(image_path):

    image = cv2.imread(image_path)
    cv2.imshow("Image originale", image)
    if image is None:
        raise FileNotFoundError("Impossible de lire l'image. Vérifiez le chemin.")
    
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    print(ratio, image.shape[0])
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Image en niveaux de gris", gray)
    
    edged = cv2.Canny(gray, 75, 200)
    
    # Recherche des contours dans l'image, et tri du plus grand au plus petit
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Parcours des contours pour trouver un polygone à 4 sommets (notre document)
    screenContour = None
    for c in contours:
        # Approximation du contour pour vérifier s'il a 4 sommets
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            screenContour = approx
            break
    
    if screenContour is None:
        raise ValueError("Aucun document à 4 côtés n'a été détecté.")
    
    # 7) Définition d'une fonction utilitaire pour ordonner les 4 points
    def order_points(pts):
        # pts est un tableau de shape (4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # on sépare les points selon leur somme (x+y) et différences
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # point haut-gauche
        rect[2] = pts[np.argmax(s)]  # point bas-droit
        rect[1] = pts[np.argmin(diff)]  # point haut-droit
        rect[3] = pts[np.argmax(diff)]  # point bas-gauche
        
        return rect

    # 8) Effectuer la perspective transform (mise à plat)
    # Convertir screenContour dans le scale d’origine (avant redimensionnement)
    screenContour = screenContour.reshape(4, 2)
    screenContour = screenContour * ratio
    
    # Ordonner les points
    rect = order_points(screenContour)
    (tl, tr, br, bl) = rect
    
    # Calcul des largeurs et hauteurs de la nouvelle image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construire la matrice de destination (points cibles)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calcul de la matrice de transformation perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Application de la matrice pour obtenir l'image "scannée"
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # 9) Optionnel : conversion en niveaux de gris et en binaire pour un aspect "scan"
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = cv2.adaptiveThreshold(warped_gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          11, 2)
    
    return warped, warped_thresh

if __name__ == "__main__":
    path_image = r'C:\Workdir\Perso\TRAVAIL\Perso\DetectionDoc\asset\image2.png'
    doc_color, doc_noir_blanc = scanner_document(path_image)
    
    # Affichage du résultat
    cv2.imshow("Document scanné (couleur)", doc_color)
    cv2.imshow("Document scanné (noir et blanc)", doc_noir_blanc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
