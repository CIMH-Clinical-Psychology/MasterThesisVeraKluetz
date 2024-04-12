nur 280 kanäle in Auswertung statt 360... ist eyetracking Kanal noch drin?
Ist eyetracking Kanal Teil von picks=meg?

ca 8 Minuten pro Participant caching first time, 4h 40 min gesamte Datei with fif and csv saving
9h für MEG_preprocessing 4s epoch

kinit alle 48 h auf Linux ausführen, um Anmeldung neu zu reaktivieren und dadurch Zugriff auf die shared files/network zu behalten

in csv 306 MEG Kanäle + 3 (condition, epoch, time)

## To-Dos:

- Einen Boxplot mit seaborn oder matplotlib ertellen, was pro Proband die durchschnittliche gif länge zeigt. Auch maxima und minima anzeigen. Auch schauen, ob Info, ob ein Knopf bei emotionaler Reaktion gedrückt wurde oder nicht, ob man die dann besser dekodieren kann
    
- später: als Trigger gif offset nehmen, tmin = -2 sek (oder je nachdem wie die durchschnittliche/max Länge ist) tmax = 0


## Fragen:
in welchem Scenario sollten sie leer sein? Wir haben doch davor schon die missing aussortiert

missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing]
for p, participant in enumerate(participants):  # (6, 7)]: # for testing purposes we might use only 1 participant, so 2 instead of 36

    if participant in ():  # these are missing
        continue



## new headline:
 our goal is to develop a method, so in which steps to build and train a classifier so that it can decode emotions.
Our goal is to investigate emotional memory replay without having to stick electrodes into a human's head.










