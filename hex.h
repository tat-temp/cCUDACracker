/*  Copyright (c) 2015 Ryan Castellucci, All Rights Reserved */
#ifndef __BRAINFLAYER_HEX_H_
#define __BRAINFLAYER_HEX_H_

#include <stdint.h>
#include <stddef.h>

/* Decode a hex string into up to bin_sz bytes. Non-hex characters (whitespace,
   newlines) are skipped, so a trailing '\n' on each line is harmless. Returns
   the number of bytes written. */
static inline int unhex(const char *hex, size_t hex_sz,
                        unsigned char *bin, size_t bin_sz) {
  size_t i, j = 0;
  int hi = -1;
  for (i = 0; i < hex_sz && j < bin_sz; ++i) {
    int v;
    char c = hex[i];
    if      (c >= '0' && c <= '9') v = c - '0';
    else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
    else continue; /* skip non-hex bytes */
    if (hi < 0) {
      hi = v;
    } else {
      bin[j++] = (unsigned char)((hi << 4) | v);
      hi = -1;
    }
  }
  return (int)j;
}

#endif /* __BRAINFLAYER_HEX_H_ */
