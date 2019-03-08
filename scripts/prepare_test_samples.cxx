#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TGenPhaseSpace.h"

Int_t N_EVENTS = 100000;
Double_t B0_MASS = 5279.58;
Double_t PION_MASS = 139.57018;


int prepare_two_body(TString filename)
{
    TFile out_file(filename, "RECREATE");
    TTree out_tree("events", "Generated events");

    TLorentzVector B(0.0, 0.0, 0.0, B0_MASS);
    Double_t masses[2] = {PION_MASS, PION_MASS};

    TGenPhaseSpace event;
    event.SetDecay(B, 2, masses);

    TLorentzVector pion_1, pion_2;
    Double_t weight;
    out_tree.Branch("pion_1", "TLorentzVector", &pion_1);
    out_tree.Branch("pion_2", "TLorentzVector", &pion_2);
    out_tree.Branch("weight", &weight, "weight/D");


    for (Int_t n=0; n<N_EVENTS; n++){
        weight = event.Generate();
        TLorentzVector *p_1 = event.GetDecay(0);
        pion_1.SetPxPyPzE(p_1->Px(), p_1->Py(), p_1->Pz(), p_1->E());
        TLorentzVector *p_2 = event.GetDecay(1);
        pion_2.SetPxPyPzE(p_2->Px(), p_2->Py(), p_2->Pz(), p_2->E());
        out_tree.Fill();
    }
    out_file.Write();
    return 0;
}


int prepare_three_body(TString filename)
{
    TFile out_file(filename, "RECREATE");
    TTree out_tree("events", "Generated events");

    TLorentzVector B(0.0, 0.0, 0.0, B0_MASS);
    Double_t masses[3] = {PION_MASS, PION_MASS, PION_MASS};

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


int prepare_four_body(TString filename)
{
    TFile out_file(filename, "RECREATE");
    TTree out_tree("events", "Generated events");

    TLorentzVector B(0.0, 0.0, 0.0, B0_MASS);
    Double_t masses[4] = {PION_MASS, PION_MASS, PION_MASS, PION_MASS};

    TGenPhaseSpace event;
    event.SetDecay(B, 4, masses);

    TLorentzVector pion_1, pion_2, pion_3, pion_4;
    Double_t weight;
    out_tree.Branch("pion_1", "TLorentzVector", &pion_1);
    out_tree.Branch("pion_2", "TLorentzVector", &pion_2);
    out_tree.Branch("pion_3", "TLorentzVector", &pion_3);
    out_tree.Branch("pion_4", "TLorentzVector", &pion_4);
    out_tree.Branch("weight", &weight, "weight/D");


    for (Int_t n=0; n<N_EVENTS; n++){
        weight = event.Generate();
        TLorentzVector *p_1 = event.GetDecay(0);
        pion_1.SetPxPyPzE(p_1->Px(), p_1->Py(), p_1->Pz(), p_1->E());
        TLorentzVector *p_2 = event.GetDecay(1);
        pion_2.SetPxPyPzE(p_2->Px(), p_2->Py(), p_2->Pz(), p_2->E());
        TLorentzVector *p_3 = event.GetDecay(2);
        pion_3.SetPxPyPzE(p_3->Px(), p_3->Py(), p_3->Pz(), p_3->E());
        TLorentzVector *p_4 = event.GetDecay(3);
        pion_4.SetPxPyPzE(p_4->Px(), p_4->Py(), p_4->Pz(), p_4->E());
        out_tree.Fill();
    }
    out_file.Write();
    return 0;
}


int prepare_test_samples(TString two_body_file, TString three_body_file, TString four_body_file){
    if (!gROOT->GetClass("TGenPhaseSpace")) gSystem->Load("libPhysics");

    prepare_two_body(two_body_file);
    prepare_three_body(three_body_file);
    prepare_four_body(four_body_file);

    return 0;
}
