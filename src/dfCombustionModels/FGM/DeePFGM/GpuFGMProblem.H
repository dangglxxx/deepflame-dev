#include "volFields.H"
namespace Foam
{
    namespace combustionModels
 {   
struct GpuFGMProblem
{

    GpuFGMProblem( ) = default;
    // GpuFGMProblem( )
    //     : h(0),z(0), c(0), gz(0),gc(0),gcz(0),cellid(0){ }

    scalar h;
    scalar z;
    scalar c;
    scalar gz;
    scalar gc;
    scalar gcz;
    label cellid;

    // TODO: implement!
    bool operator==(const GpuFGMProblem& rhs) const
    {
        return false;
    }

    bool operator!=(const GpuFGMProblem& rhs) const
    {
        return !(*this == rhs);
    }
};

//- Serialization for send
static inline Ostream& operator<<(Ostream& os, const GpuFGMProblem& p)
{

    os << p.h;
    os << p.z;
    os << p.c;
    os << p.gz;
    os << p.gc;
    os << p.gcz;
    os << p.cellid;

    return os;
}
static inline Istream& operator>>(Istream& is, GpuFGMProblem& p)
{

    is >> p.h;
    is >> p.z;
    is >> p.c;
    is >> p.gz;
    is >> p.gc;
    is >> p.gcz;
    is >> p.cellid;

    return is;
}

//- Get a serialized problem from IStream
}
}