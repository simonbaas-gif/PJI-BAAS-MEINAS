# PJI Notes :

## Binomes

 - Baas Simon
 - Meinas Julien


## Mobileye Technology

Leader dans le développement de technologies pour assistance dans la conduite.

Mobileye prend en charge une suite complète de fonctions ADAS, AEB, LDW, FCW, LKA, LC, TJA.

ADAS (Advanced Driver Assistance Systems) Basé sur un spectre passive/active.
Un système passif alerte le conducteur d'un danger potentiel. Ainsi le conducteur peut effectuer une action afin de corriger l'erreur.
Par exemple :  
            - LDW (Lane Departure Warning) Avertissement de sortie de voie. Avertit le conducteur d'un départ de voie involontaire

            - FCW (Forward Collison Warning) Avertissement de collision avant. Indique dans la dynamique actuelle par rapport au véhicule qui précède, une collision est imminente. Le conducteur doit freiner pour éviter la collision.

Au contraire, les systèmes de sécurité actif prennent la main.
AEB (Automatic Emergency Baking) Freinage d'urgence automatique, identifie la collision et les freins imminents sans aucune intervention du conducteur.
ACC (Adaptive Cruise Control) Régulateur de vitesse adaptatif.
LKA (Lane Keeping Assist) Assistant de maintien de voie.
LC (Lane Centering) Le centrage de voie.
TJA (Traffic Jam Assist) Assistant d'embouteillage.
TSR (Traffic Sign Recognition).
IHC (Intelligent High-beam Control).
Toute ces fonctions sont pris en charge à l'aide d'une seule caméra montée sur le pare-brise.

Mobileye possède un système d'avertissement qui peut être installé sur n'importe quel véhicule existant.
Mobileye se base uniquement sur les caméras.

Autonome, défi de détection :
- Espace libre, déterminer la zone de conduite
- Chemin de circulation, la géométrie des itinéraires dans la zone de conduite
- Objets en mouvement, tous les usages de la route dans la zone ou le chemin carrossable
- Sémantique des scène, vaste vocabulaire des indices visuels comme les feux de circulation et leur couleur, panneaux de signalisation, clignotants, direction du regard des piétons, marquages routiers.


## Framework open source pour détection d'objets

- TOP 10 - Traitement d'images en Python :

https://moncoachdata.com/blog/10-outils-de-traitement-dimages-en-python/

- API Google en TensorFlow :

https://www.actuia.com/actualite/google-publie-open-source-nouvelle-api-de-detection-dobjets-tensorflow/

En ce qui concerne l'API, celle ci est déjà implémenté dans produit comme NestCam, Recherches d'images ou encore Street View.

- Code Open Source TensorFlow  (série de modèles pré-entraînés):

https://github.com/tensorflow/models/tree/master/research/object_detection

- Caractéristiques des différents frameworks

https://www.developpez.com/actu/239178/Microsoft-propose-ML-NET-un-framework-d-apprentissage-automatique-open-source-qui-rend-l-apprentissage-automatique-accessible-aux-developpeurs-NET/

Au niveau des framework il en existe 3 principaux, TensorFlow, PyTorch et ML.NET.

Pour ML.NET, il suit la logique des langages .NET. C'est donc un framework Microsoft qui va pouvoir s'intégrer aux projets .NET soit des projets écrits en C#

TensorFlow, framework de Google. Le principal langage de programmation de TensorFlow est le Python, mais le C++ et le Java sont également pris en charge.
TensorFlow dispose de nombreux didacticiels, documents et projets.

PyTorch, framework de Facebook, successeur de Torch. Basé sur Python, exploite les principaux packages Python tels que NumPy. Plus simple de créer des algorithmes complexes comme un réseau de neurones récurrent.


## Librairies d'images

- Libraire Coco (PythonAPI et Librairie d'images)

https://github.com/cocodataset/cocoapi

- Site de Coco

http://cocodataset.org/#home

- Librairie ImageNet

http://image-net.org/


## Documents intéressants pour le projet :

https://arxiv.org/pdf/1902.07830.pdf

- Méthode de Viola et Jones :

https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Viola_et_Jones

- Le modèle ONNX

https://docs.microsoft.com/fr-fr/dotnet/machine-learning/tutorials/object-detection-onnx

ONNX (Open Neural Network Exchange) est un format open source pour les modèles IA. ONNX prend en charge l'interopérabilité entre les frameworks. On peut donc entraîner un modèle dans l'un des nombreux frameworks (K, ML.NET, PyTorch, ...). On peut donc par exemple passer de PyTorch à ML.NET en consommant le modèle ONNX.

- Le site ONNX

https://onnx.ai/

A noter que le Framework/Converter supporte PyTorch, TensorFlow et beaucoup d'autre framework opensource.
En fonction du framework/outil utilisé, la procédure pour convertir le projet au format ONNX est différente. Les informations concernant cette procédure sont décrites ici : https://github.com/onnx/tutorials#converting-to-onnx-format

- Tuto : Construire un modèle de reconnaissance de produits sur mesure avec TensorFlow

https://artefact.com/fr-fr/news/comment-utiliser-tensorflow-et-ses-ressources-open-source-pour-construire-un-modele-de-reconnaissance-de-produit-sur-mesure/
