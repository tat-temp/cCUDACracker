/*  Copyright (c) 2015 Ryan Castellucci, All Rights Reserved */
#ifndef __BRAINFLAYER_HASH160_H_
#define __BRAINFLAYER_HASH160_H_

#include <stdint.h>

/* A 20-byte hash160, accessible as 5 little-endian 32-bit words (ul) or 20
   bytes (uc). The bloom hash functions in bloom.h operate on the word view. */
typedef struct hash160_s {
  union {
    uint32_t ul[5];
    uint8_t  uc[20];
  };
} hash160_t;

#endif /* __BRAINFLAYER_HASH160_H_ */
