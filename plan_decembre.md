# Plan Détaillé du Rapport / Plan d'Avancement
#### Deep Learning for Autonomous Cars
#### Titouan Le Mao, Kevin Levy, Samuele Marino
#### 17 Décembre 2021

##### Lien du GitHub : https://github.com/FPoutre/Autuonomous_Cars_Raspberry_Pi
##### Lien du Rapport : https://www.overleaf.com/read/tqkdnhnydrpb

## Plan du Rapport

### Introduction
##### Identification du Projet
Finalisé.
##### Présentation du Sujet
Finalisé.
### Etat de l'Art
##### Langages Interprétés contre Langages Compilés
En cours.
Points manquants :
- Citations
- Benchmarks Python vs C++
##### Tensorflow Lite
En cours.
Points manquants :
- Citations
- Benchmarks Tensorflow vs Tensorflow Lite
##### Pruning
En cours.
Points manquants :
- TITOUAN, ATTAQUE !
##### Quantization
En cours.
Points manquants :
- TITOUAN, ATTAQUE !
### Travail Réalisé
##### Organisation
A faire.
- Repartition des Tâches
- Rythme de Travail
- Retours avec les encadrants
##### Benchmarks de Temps de Prédiction
A faire.
- En Python
- Echec en C++
- Echec avec les transpileurs Python->C++
- L'API "cross-language" TFLite
##### Optimisation des Modèles
A faire.
- Modèles Retenus
- Compilation des Modèles
- Pruning des Modèles & Ajustement des Paramètres
- Quantization des Modèles & Ajustement des Paramètres
##### Développement d'un Orchestrateur
A faire.
- Intérêt
- Architecture Logicielle
- Implémentation du Suivi de Ligne
- Implémentation de la Détection des Panneaux
### Conclusion
A faire.
- Résumé
- Réflexions sur le Travail Effectué
- Remerciements

## Produits Intermédiaires
- Script de benchmark de modèles de prédiction de trajectoire : https://github.com/FPoutre/Autuonomous_Cars_Raspberry_Pi/blob/main/Benchmarks/benchmarks.py
- Script de pruning et de quantization de modèles : https://github.com/FPoutre/Autuonomous_Cars_Raspberry_Pi/blob/main/Pruning/pruning_and_quantization.py
- Orchestrateur Python de suivi de ligne et de lecture de panneaux : https://github.com/FPoutre/Autuonomous_Cars_Raspberry_Pi/tree/main/Orchestrator
- Modèles de suivi de ligne temporaires : https://github.com/FPoutre/Autuonomous_Cars_Raspberry_Pi/tree/main/LaneFollowingModel
- Modèle de reconnaissance de panneaux : https://github.com/FPoutre/Autuonomous_Cars_Raspberry_Pi/tree/main/Real-Time-Traffic-Sign-Detection

## Bibliographie
- LTD, Raspberry Pi. Buy a Raspberry Pi 4 Model B. Raspberry Pi. URL : https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
- PiCar-X - Smart Video Robot Car Kit for Raspberry Pi ,Support Ezblock/Python Code. SunFounder. URL : https://www.sunfounder.com/products/picar-x
- TensorFlow Lite | ML pour appareils mobiles et de périphérie. TensorFlow. URL : https://www.tensorflow.org/lite?hl=fr
- M. Li, E. Yumer, et D. Ramanan, « Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints », arXiv:1905.04753 [cs], juin 2020. [En ligne]. Disponible sur: http://arxiv.org/abs/1905.04753
- Y. Gao et al., « Evaluation and Optimization of Distributed Machine Learning Techniques for Internet of Things », arXiv:2103.02762 [cs], mars 2021. [En ligne]. Disponible sur: http://arxiv.org/abs/2103.02762
- L. Giffon, S. Ayache, H. Kadri, T. Artieres, et R. Sicre, « PSM-nets: Compressing Neural Networks with Product of Sparse Matrices », in 2021 International Joint Conference on Neural Networks (IJCNN), Shenzhen, China, juill. 2021, p. 1‑8. doi: 10.1109/IJCNN52387.2021.9533408.
- T. Jin et S. Hong, « Split-CNN: Splitting Window-based Operations in Convolutional Neural Networks for Memory System Optimization », in Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, New York, NY, USA, avr. 2019, p. 835‑847. doi: 10.1145/3297858.3304038.
- M. Zhu et S. Gupta, « To prune, or not to prune: exploring the efficacy of pruning for model compression », arXiv:1710.01878 [cs, stat], nov. 2017. [En ligne]. Disponible sur: http://arxiv.org/abs/1710.01878
