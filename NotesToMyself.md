nur 280 kanäle in Auswertung statt 360... ist eyetracking Kanal noch drin?

ca 8 Minuten pro Participant caching first time, 4h 40 min gesamte Datei with fif and csv saving
9h für MEG_preprocessing 4s epoch

kinit alle 48 h auf Linux ausführen, um Anmeldung neu zu reaktivieren und dadurch Zugriff auf die shared files/network zu behalten

in csv 306 MEG Kanäle + 3 (condition, epoch, time)

## To-Dos:

- epochs per participant schöner machen und berechnen
  libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast

- Einen Boxplot mit seaborn oder matplotlib ertellen, was pro Proband die durchschnittliche gif länge zeigt. Auch maxima und minima anzeigen. Auch schauen, ob Info, ob ein Knopf bei emotionaler Reaktion gedrückt wurde oder nicht, ob man die dann besser dekodieren kann
    
- später: als Trigger gif offset nehmen, tmin = -2 sek (oder je nachdem wie die durchschnittliche/max Länge ist) tmax = 0


## Fragen:
plot_epochs_per_participant
barplot mit seabron, gap oder width oder natural_... nicht einstellbar, warum?

random forest: Wleche Parameter noch zusätzlich einstellen?

Preprocessing anschauen/vergleichen ob richtig durchgeführt
wenig preprocessed data accuracy vergleichen mit normaler

- BCI post anschauen

- - von klipscalc migrieren

## discussion / thesis points:
 our goal is to develop a method, so in which steps to build and train a classifier so that it can decode emotions.
Our goal is to investigate emotional memory replay without having to stick electrodes into a human's head.

we see that there is a 40% classification ability based on eyemovements, even after removing eog channel and performing ICA eog component rejection. Therefore, we have to make sure, that the emotion classifier in the end does not just decode corresponding eye movement. Do e.g. negative emotions create another gaze pattern than positive emotions?


idea: put GIFs in Ai video to text model and get internal representation (Samu showed me a demo on huggingface)
https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb
Then use MEG data and internal representation to find/get features or correlations etc


## Plan:
use Logistic Regression for everything:
1. only EOG data: preprocessing? Yes, some steps needed
2. Meg Data without ICA EOG filter, without EOG data: create data that is not ICA filtered, then let it run with nice plot and resampling
3. MEG Data with ICA EOG filter, without EOG data: make a nicer plot, let it run without resampling; also let it run with random forest for comparison

Do everything with data that has been less preprocessed??? Kinda raw data that only has had: high pass filtering and bad channel interpolation (read up on Simon's paper)


-------------------------------
plot epochs per participant vom Abstand und den Farben nicer machen,  für epochs die tatsächlich verwendet werden

------------------------------








