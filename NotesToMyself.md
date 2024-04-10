nur 280 kanäle in Auswertung statt 360... ist eyetracking Kanal noch drin?
Ist eyetracking Kanal Teil von picks=meg?

ca 8 Minuten pro Participant caching first time, 4h 40 min gesamte Datei with fif and csv saving

decide which environments to use (sckit learn pipelines?)- ask Alpay 
kinit alle 48 h auf Linux ausführen, um Anmeldung neu zu reaktivieren und dadurch Zugriff auf die shared files/network zu behalten
PUG, Bahn 240€ Hin- und Rückfahrt Mi bis So, 60-70€ p.P. pro Tag (sehr günstig für da), 110€ Konferenz Gebühr

in csv 306 MEG Kanäle + 3 (condition, epoch, time)

## To-Dos:

- Einen Boxplot mit seaborn oder matplotlib ertellen, was pro Proband die durchschnittliche gif länge zeigt. Auch maxima und minima anzeigen. Auch schauen, ob Info, ob ein Knopf bei emotionaler Reaktion gedrückt wurde oder nicht, ob man die dann besser dekodieren kann

- zuerst: gif onset als eventID nehmen, dann mit ML dekodieren in welchem quadranten das gif gezeigt wurde (sanity check) (ohne EOG und bei ICA 1. Komponente (auch EOG) rausfiltern). Dafür eine Funktion schreiben, die mir für jede Probandennummer (Input) eine Liste aller Gifs rausgibt und in welchem Quadranten das Gif gezeigt wurde (Lösung)
    
- später: als Trigger gif offset nehmen, tmin = -2 sek (oder je nachdem wie die durchschnittliche/max Länge ist) tmax = 0


ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:

## Fragen:

## new headline:
 our goal is to develop a method, so in which steps to build and train a classifier so that it can decode emotions.
Our goal is to investigate emotional memory replay without having to stick electrodes into a human's head.










