# Targeted-attacks-on-malicious-traffic-detection
## Introduction:
This is a project about a targeted attacks method for malicious traffic detection.

This work has been submitted to **ICASSP2024**. 

We submitted part of the project, and the full project will be published after admission.

## Abstract:
Leveraging deep learning to detect malicious network traffic is a crucial technology in network management and network security. However, deep learning security has raised concerns among scholars.
In this work, we explore executing targeted adversarial attacks for multi-classification malicious traffic detection with limited interactions. 
Specifically, we constrain the number of interactions with detection and employ a hop-skip-jump attack (HSJA) to generate a small number of adversarial samples.
These adversarial samples are then heuristically used to train a generative adversarial network (GAN) to generate a substantial quantity of adversarial samples.
Experiments demonstrate that our method is more adversarial and displays a certain degree of generalization compared with other methods.

## Script Description:

  IDS.py
      A malicious traffic training model framework, you can add arbitrary models. Used for subsequent adversarial attack testing.
      
  HSJA.py
      Employing a hop-skip-jump attack (HSJA) to generate a small number of adversarial samples.
