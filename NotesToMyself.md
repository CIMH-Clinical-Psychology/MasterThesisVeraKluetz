nur 280 kanäle in Auswertung statt 360... ist eyetracking Kanal noch drin?

ca 8 Minuten pro Participant caching first time, 4h 40 min gesamte Datei with fif and csv saving
9h für MEG_preprocessing 4s epoch

kinit alle 48 h auf Linux ausführen, um Anmeldung neu zu reaktivieren und dadurch Zugriff auf die shared files/network zu behalten

in csv 306 MEG Kanäle + 3 (condition, epoch, time)

## To-Dos:

- Einen Boxplot mit seaborn oder matplotlib ertellen, was pro Proband die durchschnittliche gif länge zeigt. Auch maxima und minima anzeigen. Auch schauen, ob Info, ob ein Knopf bei emotionaler Reaktion gedrückt wurde oder nicht, ob man die dann besser dekodieren kann
    
- später: als Trigger gif offset nehmen, tmin = -2 sek (oder je nachdem wie die durchschnittliche/max Länge ist) tmax = 0


## Fragen:
plot_epochs_per_participant
barplot mit seabron, gap oder width oder natural_... nicht einstellbar, warum?



## new headline:
 our goal is to develop a method, so in which steps to build and train a classifier so that it can decode emotions.
Our goal is to investigate emotional memory replay without having to stick electrodes into a human's head.




## Plan:
use Logistic Regression for everything:
1. only EOG data: preprocessing? Yes, some steps needed
2. Meg Data without ICA EOG filter, without EOG data: create data that is not ICA filtered, then let it run with nice plot and resampling
3. MEG Data with ICA EOG filter, without EOG data: make a nicer plot, let it run without resampling

Do everything with data that has been less preprocessed??? Kinda raw data that only has had: high pass filtering and bad channel interpolation (read up on Simon's paper)







