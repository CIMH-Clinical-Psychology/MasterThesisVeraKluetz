decide which environments to use (sckit learn pipelines?)- ask Alpay 


kinit alle 48 h auf Linux ausführen, um Anmeldung neu zu reaktivieren und dadurch Zugriff auf die shared files/network zu behalten

## To-Dos:
- Einen Boxplot mit seaborn oder matplotlib ertellen, was pro Proband die durchschnittliche gif länge zeigt. Auch maxima und minima anzeigen. Auch schauen, ob Info, ob ein Knopf bei emotionaler Reaktion gedrückt wurde oder nicht, ob man die dann besser dekodieren kann

- zuerst: gif onset als eventID nehmen, dann mit ML dekodieren in welchem quadranten das gif gezeigt wurde (sanity check) (ohne EOG und bei ICA 1. Komponente (auch EOG) rausfiltern). Dafür eine Funktion schreiben, die mir für jede Probandennummer (Input) eine Liste aller Gifs rausgibt und in welchem Quadranten das Gif gezeigt wurde (Lösung). Dazu diese Info aus .csv Datei mit pandas rauslesen (csv liegt in participantdata folder). 
    
- später: als Trigger gif offset nehmen, tmin = -2 sek (oder je nachdem wie die durchschnittliche/max Länge ist) tmax = 0


## Fragen:
- epochs.plot(show=False) -> manchmal gibt es epochs, bei denen das Signal 0 ist, kein 'graph' in dieser Epoche vorhanden in keinem einzigen channel (bei Participant 1 deutlich häufiger als bei Par 2)
- find events: die events sollten doch bei allen stimulus sein, für uns interessant: wenn ich sie plotte dann nach allen triger_gif_onset -> ist so aber nicht, kein Zusammenhang zwischen stimulus trigger und extrahierten (geplotteten) events erkennbar
  -> Was genau plotte ich dann, wenn ich events plotte?

# rausgefunden:
`epochs.resample`:
   - This method applies an anti-aliasing filter to prevent aliasing artifacts, which can occur when the signal contains frequencies above the Nyquist frequency after resampling.
   - bigger choice of resampling factor, set sampling rate to an arbitrary value or resample to a non-integer factor.
   - Higher computational cost compared to `epochs.decimate`.

`epochs.decimate`:
   - It reduces the number of samples by keeping every `decim`-th sample and discarding the others.
   - It doesn't apply an anti-aliasing filter by default
   - simple and quick downsampling operation by an integer factor.






