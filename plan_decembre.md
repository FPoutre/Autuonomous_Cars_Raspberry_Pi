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