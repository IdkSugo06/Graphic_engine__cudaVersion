#pragma once

#include "Rasterization functions.cuh"
#include "Primitive assembler.cuh"
#include "Clipping functions.cuh"
#include "Reset functions.cuh"
#include "Shaders.cuh"
#include "PostProcessing.cuh"


//! ATTENZIONE: i kernel lanciati non si eseguiranno se un errore avviene durante l'esecuzione di un kernel precedente
//! ATTENZIONE: in alcuni casi, i kernel possono essere troppo pesanti (operazioni) e non venir eseguiti (i kernel successivi verranno comunque eseguiti, ma nessuna modifica verrà effettuata dal kernel troppo pesante)
//! tips:
//! eventuali modifiche applicate dal kernel agli argomenti passati verranno scartate non appena il kernel incontra un errore