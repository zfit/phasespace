#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TGenPhaseSpace.h"

Int_t N_EVENTS = 1000000;

int bench_tgenphasespace()
{
    TLorentzVector B(0.0, 0.0, 0.0, 5279.0);
    Double_t masses[3] = {139.6, 139.6, 139.6};

    if (!gROOT->GetClass("TGenPhaseSpace")) gSystem->Load("libPhysics");
    TGenPhaseSpace event;
    event.SetDecay(B, 3, masses);

    for (Int_t n=0; n<N_EVENTS; n++) event.Generate();
    return 0;
}
