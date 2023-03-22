// File.cpp
//
// JHelas (c) 11/07
//

#include <unistd.h>

#include "File.h"


CFile::CFile()
  : m_hFile(FILE_NULL) {
 }


CFile::CFile(PCSTR sFileName, int nOpenFlags, mode_t nMode /*= 0640*/)
  : m_hFile(FILE_NULL) {

  CFileException  e;

  if (! Open(sFileName, nOpenFlags, nMode, &e)) {
    throw new CFileException;
   }
 }


CFile::~CFile() {
  Close();
 }


bool CFile::Open(PCSTR sFileName, int nOpenFlags, mode_t nMode /*= 0640*/, CFileException *pException /*= NULL*/) {
  bool bRet(false);

  Close();

  int hFile = ::open(sFileName, nOpenFlags, nMode);

  if (hFile == FILE_NULL) { // Error ?
    if (pException != NULL) {
      *pException = CFileException();
     }
   }
   else {
    m_hFile = hFile;
    bRet    = true;
   }

  return bRet;
 }


void CFile::Close() {
  if (m_hFile != FILE_NULL) {
    int nError = ::close(m_hFile);

    m_hFile = FILE_NULL;

    if (nError == FILE_ERROR) throw new CFileException();
   }
 }


ssize_t CFile::Read(void *pBuffer, size_t nCount) {
  if (nCount == 0) return 0;

  ssize_t nRead = ::read(m_hFile, pBuffer, nCount);

  if (nRead == FILE_ERROR) throw new CFileException();

  return nRead;
 }


off_t CFile::Seek(off_t nOffset, int nFrom) {
  off_t nPosition(::lseek(m_hFile, nOffset, nFrom));

  if (nPosition == FILE_ERROR) throw new CFileException();

  return nPosition;
 }
