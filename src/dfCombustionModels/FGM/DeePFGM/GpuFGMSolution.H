#include "volFields.H"
namespace Foam
{
    namespace combustionModels
 {   
struct GpuFGMSolution
{

    GpuFGMSolution() = default;

    // GpuFGMSolution(label nomega)
    //     : omega(nomega, 0.0)
    // {
    // }

    bool operator==(const GpuFGMSolution& rhs) const
    {
        return false;
    }

    bool operator!=(const GpuFGMSolution& rhs) const
    {
        return !(*this == rhs);
    }

    scalar omegac;
    scalar comegac;
    scalar zomegac;
    label cellid;
};

//- Serialization for send
static inline Ostream& operator<<(Ostream& os, const GpuFGMSolution& s)
{
    os << s.omegac;
    os << s.comegac;
    os << s.zomegac;
    os << s.cellid;
    return os;
}

//- Get a serialized solution from IStream
static inline Istream& operator>>(Istream& is, GpuFGMSolution& s)
{
    is >> s.omegac;
    is >> s.comegac;
    is >> s.zomegac;
    is >> s.cellid;
    return is;
}

 }
}