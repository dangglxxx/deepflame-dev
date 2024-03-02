/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::combustionModels::DeePFGM

Description
    Partially stirred reactor turbulent combustion model.

    This model calculates a finite rate, based on both turbulence and chemistry
    time scales.  Depending on mesh resolution, the Cmix parameter can be used
    to scale the turbulence mixing time scale.

SourceFiles
    DeePFGM.C

\*---------------------------------------------------------------------------*/

#ifndef DeePFGM_H
#define DeePFGM_H

#include "baseFGM.H"
#include "tableSolver.H"
#include "volFields.H"
#include "GpuFGMProblem.H"
#include "GpuFGMSolution.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{

/*---------------------------------------------------------------------------*\
                            Class DeePFGM Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionThermo>
class DeePFGM
:
    public baseFGM<ReactionThermo>,
    public tableSolver
{

public:

    //- Runtime type information
    TypeName("DeePFGM");


    //- Runtime type information


    // Constructors

        //- Construct from components
        DeePFGM
        (
            const word& modelType,
            ReactionThermo& thermo,
            const compressibleTurbulenceModel& turb,
            const word& combustionProperties
        );

        //- Disallow default bitwise copy construction
        DeePFGM(const DeePFGM&);


    //- Destructor
    virtual ~DeePFGM();


    // Member Operators

        //- Correct combustion rate
        virtual void correct();

        //- retrieval data from table
        virtual void retrieval( );

        pybind11::object my_module;
        pybind11::object model;
        pybind11::object phimax;
        pybind11::object phimin;
        pybind11::object xmax;
        pybind11::object xmin;
        virtual void getGPUFGMProblemCells(Foam::DynamicList<GpuFGMProblem>& GPUFGMproblemList);
        virtual void getDNNinputs
        (
            const Foam::DynamicBuffer<GpuFGMProblem>& problemBuffer,
            std::vector<label>& outputLength,
            std::vector<std::vector<double>>& DNNinputs,
            std::vector<Foam::DynamicBuffer<label>>& cellIDBuffer,
            std::vector<std::vector<label>>& problemCounter
        );
        virtual void updateSolutionBuffer
        (
            Foam::DynamicBuffer<GpuFGMSolution>& solutionBuffer,
            const std::vector<std::vector<double>>& results,
            const std::vector<Foam::DynamicBuffer<Foam::label>>& cellIDBuffer,
            std::vector<std::vector<Foam::label>>& problemCounter
        );
        //- Disallow default bitwise assignment
        void operator=(const DeePFGM&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "DeePFGM.C"
#endif

#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// ************************************************************************* //
