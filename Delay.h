// Delay.h
//
//  JHelas (c) 11/06
//

#ifndef DELAY_H_INCLUDED
#define DELAY_H_INCLUDED

#include <time.h>

typedef const char *    PCSTR;


class CDelay
{
public:
// Construction
  CDelay(PCSTR sFormat = NULL)
    : m_sFormat(sFormat),
      m_nStart(::clock()) { }

  CDelay(const CDelay &Obj);

  virtual ~CDelay() {
    double fDuration = (double) (::clock() - m_nStart) / CLOCKS_PER_SEC;

    if (m_sFormat == NULL) {
      printf("CDelay::~CDelay()--> Duration: %2.2f\n", fDuration);
     }
     else {
      printf(m_sFormat, fDuration);
     }
   }


// Attributes
private:
  PCSTR             m_sFormat;
  clock_t           m_nStart;

public:

// Functions
private:

protected:

public:
  const CDelay &    operator=(const CDelay &Obj);
 };

#endif // DELAY_H_INCLUDED
