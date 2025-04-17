#### 250414
iniziato a lavorare alla parte di dataset prendendo il codice di CMDA, in particolare `dsec.py`. 

Trovata repo `dsec_det`. Forse più utile per la object detection. dato che 

#### 250415
lavorato all'adattamento del dataset per object detection
Le bounding box hanno gli stessi timestamp dei frame. Essendo leggere, si potrebbero comunque caricare tutte le bb in memoria.
Lui usa le warp e genera un train/test split andando a generare un un dizionario del tipo [image event_index] dove event index dovrebbe essere l'ultimo indice degli eventi generati fino a quell'istante. e.g. la prima riga contiene la prima immagine con l'ultimo idx dell'evento generato al suo interno.

#### 250417
Punto della situazione: Occorre caricare in memoria le bounding box. Per fare ciò, occorre prima usare il timestamp per andare a trovare tutte le bounding box all'interno del frame.

Per ora penso di caricare tutto in memoria, metto un grande todo e vediamo.

Attenzione label/not-label split. Nel test set labelled c'è data augmentation, nel train no.

Diciamo che forse conviene vedere parte di data loading da dsec-det -> tutto/maggiorparte testato, precompilato con numba. quindi ottimo lavoro. Più ordinato. Domani scrivo e vediamo che mi dicono.