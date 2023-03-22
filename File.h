// File.h
//

#ifndef FILE_H_INCLUDED
#define FILE_H_INCLUDED

#include <stdio.h>
#include <errno.h>
#include <fcntl.h>

typedef unsigned int    UINT;
typedef const char *    PCSTR;

#ifndef ssize_t
#include <sys/types.h>
#endif

/* Open flags (selection):

  O_RDONLY
  O_WRONLY
  O_RDWR
  O_APPEND
  O_CREAT
  O_TRUNC
  O_EXCL
*/

class CFileException {
public:
  CFileException()
    : m_nError(errno) { }
  virtual ~CFileException() { }

private:
  int               m_nError;

public:
  void              Delete() {
    delete this;
   }

  virtual int       GetError() const              { return m_nError; }
  void              PrintError(PCSTR sMessage) {
    errno = m_nError;

    ::perror(sMessage);
   }
 };


class CFile {
public:
  CFile();
  CFile(PCSTR sFileName, int nOpenFlags, mode_t nMode = 0640);

  virtual ~CFile();

  enum {
    FILE_NULL   = -1,
    FILE_ERROR  = -1
   };

// Attributes
private:
  int               m_hFile;

public:

// Functions
private:

protected:

public:
  virtual bool      Open(PCSTR sFileName, int nOpenFlags, mode_t nMode = 0640, CFileException *pException = NULL);
  virtual void      Close();

  ssize_t           Read(void *pBuffer, size_t nCount);

                    // nFrom := SEEK_SET, SEEK_CUR, SEEK_END
  virtual off_t     Seek(off_t nOffset, int nFrom);
 };

#endif // FILE_H_INCLUDED
