decide which environments to use (sckit learn pipelines?)- ask Alpay 


kinit alle 48 h auf Linux ausführen, um Anmeldung neu zu reaktivieren und dadurch Zugriff auf die shared files/network zu behalten


1. bandpass filter
2. EEG: es wird ge-re-referenced mit REST (Average ü, u.A. damit noise, dass den ganzen Scalp betrifft, herausgefiltert (subtrahiert) werden kann. Aber bei PowerLine noise wird es verstärkt, da es ungleichmäßig die Scalp Regionen beeinflusst. Diese Argumentation trifft doch wahrscheinlich auf fast alle Umwelt- und internen Faktoren zu, also ist obiges Argument doch wieder entkräftet

more line noise in REST and average (AV):

Interesting observation! However, the line noise should be common in all electrodes, so why does it get amplified when applying the average/REST reference?

That is true! Re-referencing should get rid of a lot of the 50 Hz noise (under the implication that the noise phase is the same at each electrode, which should be the case). If you look at unreferenced data, there should be much more 50 Hz noise.



Filtering Antwort falsch: kein kleiner gleich
 B < f s / 2     (sample rate: fs, signal: B)

(resampling mit upper bounf of 100Hz) As far as I understand, we should be able to see the maximum frequency of 50 Hz as according to the Nyquist rule, the sampling frequency must be at least (NO, IT MUST BE GREATER; EQUAL IS NOT ENOUGH) twice the highest frequency that we wish to analyse. Any higher frequencies than 50 Hz (NO, IT INCLUDES 50 HZ) might show aliasing where sampling is not fast enough to construct an accurate waveform record.
