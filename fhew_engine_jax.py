import math

import jax.numpy as jnp
import jax.random as random
import jax
jax.config.update("jax_enable_x64", True)

class fhew_engine:
    def __init__(self, device=None):
        self.stddev = 3.2
        self.logQ = 27
        self.logQks = 14

        self.N = 1 << 10
        self.Nh = self.N >> 1
        self.Q = 1 << self.logQ
        self.Qks = 1 << self.logQks

        self.n = 571
        self.q = 2*self.N

        self.d = 2
        self.logB = 9

        assert self.d * self.logB <= self.logQ

        self.dks = 2
        self.logBks = 7

        self.Bks = 1 << self.logBks

        assert self.dks * self.logBks <= self.logQks

        print("Using jax.numpy")

        self.key = random.PRNGKey(0)  # Initialize a PRNGKey for random number generation

        self.f_nand = jnp.zeros([self.N], dtype=jnp.int32)

        for i in range(self.N):
            self.f_nand = self.f_nand.at[i].set(self.nand_map(-i))

        self.root_powers = jnp.exp((1j * math.pi / self.N) * jnp.arange(self.Nh))

        self.root_powers_inv = jnp.exp((1j * math.pi / self.N) * jnp.arange(0, -self.Nh, -1))

        # parameters for decomposition
        # we will use B-ary decomposition, i.e., cut digits by logB bits
        self.decomp_shift = self.logQ - self.logB * jnp.arange(self.d, 0, -1).reshape(self.d, 1)
        self.mask = (1 << self.logB) - 1

        self.gvector = 1 << self.decomp_shift

        # for signed decomposition
        self.msbmask = 0
        for shift in self.decomp_shift:
            self.msbmask += (1 << (shift[0] + self.logB - 1))

        # parameters for LWE key switching decomposition
        self.decomp_shift_ks = self.logQks - self.logBks * jnp.arange(self.dks, 0, -1).reshape(self.dks, 1)
        self.mask_ks = jnp.array([(1 << self.logBks) - 1], dtype=jnp.int32)

        self.gvector_ks = 1 << self.decomp_shift_ks
        self.decomp_shift_ks = self.decomp_shift_ks.reshape(self.dks, 1)

        self.alphapoly = self.precompute_alpha()
        self.betapoly = self.precompute_beta()

        self.brk = None
        self.ksk = None

    def create_secret_key(self):
        """ Generate secret key """
        return self.keygen(self.n)

    def create_bootstrap_key(self, sk):
        """ Generate bootstrapping keys, and return a tuple brk (blind rotation keys)
            and ksk (LWE key switching key)

            Keyword arguments:
            sk -- secret key
        """
        skN = self.keygen(self.N)

        skNfft = self.negacyclic_fft(skN)
        brk = self.brkgen(sk, skNfft)

        ksk = self.LWEkskGen(sk, skN)
    
        return brk, ksk

    def keygen(self, dim):
        """ Generate uniform binary secret of size n.
            Note: insecure PRNG is used.
        """
        self.key, subkey = random.split(self.key)  # Split the PRNGKey to get a new one for this function
        return random.randint(subkey, (dim,), minval=0, maxval=2, dtype=jnp.int32)

    def errgen(self, dim):
        """ Generate Gaussian noise of size `dim`.
            Note: insecure PRNG is used.
        """
        self.key, subkey = random.split(self.key)
        e = jnp.round(self.stddev * random.normal(subkey, (dim,)))
        e = e.squeeze()
        return e.astype(jnp.int32)

    def uniform(self, dim, modulus):
        """ Generate uniform noise of size `dim` mod `modulus`.
            Note: insecure PRNG is used.
        """
        self.key, subkey = random.split(self.key)
        return random.randint(subkey, (dim,), minval=0, maxval=modulus, dtype=jnp.int32)

    def negacyclic_fft(self, a):
        """ Negacyclic FFT on the given array a. """
        acomplex = a.astype(jnp.complex128)

        a_precomp = (acomplex[..., :self.Nh] + 1j * acomplex[..., self.Nh:]) * self.root_powers
        return jnp.fft.fft(a_precomp)

    def negacyclic_ifft(self, A):
        """ Negacyclic inverse FFT on the given array A. """
        b = jnp.fft.ifft(A)
        b *= self.root_powers_inv

        a = jnp.concatenate((b.real, b.imag), axis=-1).astype(int).astype(jnp.int32)
        # only when Q is a power-of-two
        a &= self.Q - 1

        return a

    def to_signed(self, v, logQ):
        """ Convert unsigned array to signed array mod Q.

            Keyword arguments:
            v -- unsigned array to convert
            logQ -- log_2 of Q, i.e., Q = 1 << logQ
        """
        Q = (1 << logQ)
        v &= (Q-1)
        msb = (v & Q//2) >> (logQ - 1)
        v -= Q * msb
        return v

    def decompose(self, a):
        """ Unsigned digit (B) decomposition of given array a. """
        return (a.reshape(1, *a.shape) >> self.decomp_shift.reshape(self.d, 1, 1)) & self.mask
        
    def signed_decompose(self, a):
        """ Signed decomposition of given array a.
            Uses `decompose` two times.
        """
        da = self.decompose(a + (a & self.msbmask))
        da -= self.decompose((a & self.msbmask))
        return da

    def encrypt_rgsw_fft(self, z, skfft):
        """ Return RGSW_sk(z). """
        rgsw = jnp.zeros((self.d, 2, 2, self.N), dtype=jnp.int32)

        # generate the 'a' part
        self.key, subkey = random.split(self.key)
        rgsw = rgsw.at[:, :, 1, :].set(random.randint(subkey, (self.d, 2, self.N), minval=0, maxval=self.Q, dtype=jnp.int32))

        # add error on b
        self.key, subkey = random.split(self.key)
        rgsw = rgsw.at[:, :, 0, :].set(jnp.round(self.stddev * random.normal(subkey, (self.d, 2, self.N))).astype(jnp.int32))

        rgsw = self.to_signed(rgsw, self.logQ)

        rgswfft = self.negacyclic_fft(rgsw)

        rgswfft = rgswfft.at[:, :, 0, :].set(rgswfft[:, :, 0, :] - rgswfft[:, :, 1, :] * skfft.reshape(1, 1, self.Nh))

        gzfft = self.negacyclic_fft(self.gvector * z)
        rgswfft = rgswfft.at[:, 0, 0, :].set(rgswfft[:, 0, 0, :] + gzfft)
        rgswfft = rgswfft.at[:, 1, 1, :].set(rgswfft[:, 1, 1, :] + gzfft)

        return rgswfft

    def rgswmult(self, ctfft, rgswfft):
        """ Multiply ctfft (RLWE) and rgswfft (RGSW) and return an RLWE ciphertext. """

        ct = self.negacyclic_ifft(ctfft)
        dct = self.signed_decompose(ct)
        multfft = self.negacyclic_fft(dct).reshape(self.d, 2, 1, self.Nh) * rgswfft
        return jnp.sum(multfft, axis=(0, 1))
    
    def encryptLWE(self, message, sk):
        """ Encrypt a bit.

        Keyword arguments:
        message -- a bit, 0 or 1
        sk -- secret key to use
        """

        ct = self.uniform(self.n + 1, self.q)

        ct = ct.at[0].set(0)

        ct = ct.at[0].set(message * self.q // 4 - jnp.sum(ct[1:] * sk))
        ct = ct.at[0].add(self.errgen(1))
        ct &= (self.q - 1)

        return ct

    def decryptLWE(self, ct, sk, ksk=None):
        """ Decrypt LWE ciphertext. Do key switch if needed. """

        if len(ct) > self.n + 1:
            if self.ksk is None and ksk is None:
                print("Cannot key switch to small key. Provide switching key.")
            ct_ms = (ct * (self.Qks/self.Q)).astype(jnp.int32)
            ct_ks = self.LWEkeySwitch(ct_ms, self.ksk)
            ct_n = (ct_ks * (self.q/self.Qks)).astype(jnp.int32)
        else:
            ct_n = ct

        m_dec = ct_n[0] + jnp.sum(ct_n[1:] * sk)
        m_dec %= self.q

        m_dec = jnp.round(m_dec / (self.q / 4.0)).astype(int)

        return m_dec % 4

    def extract(self, ctRLWE):
        """ Extract an LWE ciphertext from ctRLWE, holding its the constant term as plaintext."""

        beta = ctRLWE[0][0]

        alpha = ctRLWE[1, :]
        alpha = alpha.at[1:].set(-alpha[1:][::-1])

        # Convert beta to a JAX array before concatenation
        beta_array = jnp.array([beta])

        return jnp.concatenate((beta_array, alpha))


    def decompose_ks(self, a):
        """ Decompose LWE ciphertext for LWE key switching. """
        assert len(a.shape) == 1
        res = (a.reshape(1, -1) >> self.decomp_shift_ks) & self.mask_ks
        return res

    def LWEkskGen(self, sk, skN):
        """ Generate key switching key from skN to sk. """

        self.key, subkey = random.split(self.key)
        ksk = jnp.array(random.randint(subkey, (self.dks, self.Bks, self.N, self.n + 1), minval=0, maxval=self.Qks, dtype=jnp.int32))
        self.key, subkey = random.split(self.key)
        ksk = ksk.at[..., 0].set(jnp.round(self.stddev * random.normal(subkey, (self.dks, self.Bks, self.N))).astype(jnp.int32))
        ksk = ksk.at[..., 0].set(ksk[..., 0] - jnp.sum(ksk[..., 1:] * sk, axis=-1))
        ksk = ksk.at[..., 0].set(ksk[..., 0] + (self.gvector_ks * jnp.arange(self.Bks)).reshape(self.dks, self.Bks, 1) * skN)
        ksk &= (self.Qks - 1)

        return ksk

    def LWEkeySwitch(self, ctLWE, kskLWE):
        """ Switch key of ctLWE (dimension N) to a smaller key (dimension n). """

        alpha = ctLWE[1:]
        dalpha = self.decompose_ks(alpha)
        switched = jnp.zeros(self.n + 1, dtype=jnp.int32)
        switched = switched.at[0].set(ctLWE[0])

        for r in range(self.dks):
            for i in range(self.N):
                switched = switched + kskLWE[r, dalpha[r, i], i]

        switched &= (self.Qks - 1)

        return switched

    def brkgen(self, sk, skNfft):
        """ Generate blind rotation keys. """

        zero_poly = jnp.zeros([self.N], dtype=jnp.int32)

        one_poly = jnp.zeros([self.N], dtype=jnp.int32)
        one_poly = one_poly.at[0].set(1)

        brk = jnp.zeros((self.n, self.d, 2, 2, self.Nh), dtype=jnp.complex128)
        for i in range(self.n):
            if sk[i] == 0:
                brk = brk.at[i].set(self.encrypt_rgsw_fft(zero_poly, skNfft))
            else:
                brk = brk.at[i].set(self.encrypt_rgsw_fft(one_poly, skNfft))

        return brk

    def precompute_alpha(self):
        """ Precompute fft(X^i - 1), return alphapoly where alphapoly[i] = fft(X^i - 1). """

        alphapoly = jnp.zeros((self.q, self.Nh), dtype=jnp.complex128)

        for i in range(self.q):
            poly = jnp.zeros([self.N], dtype=jnp.int32)
            poly = poly.at[0].set(-1)
            if i < self.N:
                poly = poly.at[i].add(1)
            else:
                poly = poly.at[i - self.N].add(-1)
            alphapoly = alphapoly.at[i].set(self.negacyclic_fft(poly))
            
        return alphapoly

    def precompute_beta(self):
        """ Precompute fft(X^i), return alphapoly where betaapoly[i] = fft(X^i). """

        betapoly = jnp.zeros((self.q, self.Nh), dtype=jnp.complex128)

        for i in range(self.q):
            poly = jnp.zeros([self.N], dtype=jnp.int32)
            if i < self.N:
                poly = poly.at[i].set(1)
            else:
                poly = poly.at[i - self.N].set(-1)
            betapoly = betapoly.at[i].set(self.negacyclic_fft(poly))
            
        return betapoly

    def nand_map(self, i):
        """ Precompute LUT function for NAND gate. """

        i += 2 * self.N 
        i %= 2 * self.N
        if 3 * (self.q >> 3) <= i < 7 * (self.q >> 3):
            return -(self.Q >> 3)
        else:
            return self.Q >> 3 

    def gate_bootstrapping(self, ct0, ct1, brk, ksk, gate="NAND"):
        """ Do a gate bootstrapping for given `ct0` and `ct1`.
        Only NAND is supported currently.

        Keyword arguments:
        ct0, ct1 -- input ciphertexts
        brk -- blind rotation keys to use
        ksk -- LWE key switching keys to use
        gate :String -- gate to perform e.g., "NAND". 
        """

        ctsum = ct0 + ct1

        # NAND is supported only
        assert gate == "NAND"
        
        if len(ctsum) > self.n + 1:
            ctsum &= (self.Q - 1)
            ctsum_ms = (ctsum * (self.Qks/self.Q)).astype(jnp.int32)
            ctsum_ks = self.LWEkeySwitch(ctsum_ms, ksk)
            ctsum = (ctsum_ks * (self.q/self.Qks)).astype(jnp.int32)

        ctsum &= (self.q - 1)

        acc = jnp.zeros([2, self.N], dtype=jnp.int32)
        acc = acc.at[0].set(self.f_nand)

        accfft = self.negacyclic_fft(acc)

        beta = ctsum[0]
        accfft *= self.betapoly[beta]

        alpha = ctsum[1:]
        for i in range(self.n):
            ai = alpha[i]
            accfft = accfft + self.alphapoly[ai] * self.rgswmult(accfft, brk[i])

        acc = self.negacyclic_ifft(accfft)
        accLWE = self.extract(acc)
        accLWE = accLWE.at[0].add(self.Q >> 3)
        accLWE &= (self.Q - 1)
        
        return accLWE

    def setKeySwitchKey(self, ksk):
        """ Enroll LWE key switching keys to engine for simpler codes. """

        self.ksk = ksk
        return self

    def setBlindRotationKey(self, brk):
        """ Enroll blind rotation keys to engine for simpler codes. """

        self.brk = brk
        return self
    
    def nand(self, ct0, ct1):
        """ Do a NAND gate on two input ciphertexts ct0 and ct1 """

        if self.ksk is None or self.brk is None:
            print("Need to set blind rotation key and key switch key")
            print("using methods setBlindRotationKey and setKeySwitchKey.")
            return
        
        return self.gate_bootstrapping(ct0, ct1, self.brk, self.ksk)
