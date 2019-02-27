#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TGenPhaseSpace.h"

Int_t N_EVENTS = 100000;

int prepare_test_samples()
{
    TFile out_file("../data/bto3pi.root", "RECREATE");
    TTree out_tree("events", "Generated events");

    TLorentzVector B(0.0, 0.0, 0.0, 5279.0);
    Double_t masses[3] = {139.6, 139.6, 139.6};

    if (!gROOT->GetClass("TGenPhaseSpace")) gSystem->Load("libPhysics");
    TGenPhaseSpace event;
    event.SetDecay(B, 3, masses);

    TLorentzVector pion_1, pion_2, pion_3;
    Double_t weight;
    out_tree.Branch("pion_1", "TLorentzVector", &pion_1);
    out_tree.Branch("pion_2", "TLorentzVector", &pion_2);
    out_tree.Branch("pion_3", "TLorentzVector", &pion_3);
    out_tree.Branch("weight", &weight, "weight/D");


    for (Int_t n=0; n<N_EVENTS; n++){
        weight = event.Generate();
        TLorentzVector *p_1 = event.GetDecay(0);
        pion_1.SetPxPyPzE(p_1->Px(), p_1->Py(), p_1->Pz(), p_1->E());
        TLorentzVector *p_2 = event.GetDecay(1);
        pion_2.SetPxPyPzE(p_2->Px(), p_2->Py(), p_2->Pz(), p_2->E());
        TLorentzVector *p_3 = event.GetDecay(2);
        pion_3.SetPxPyPzE(p_3->Px(), p_3->Py(), p_3->Pz(), p_3->E());
        out_tree.Fill();
    }
    out_file.Write();
    return 0;
}
