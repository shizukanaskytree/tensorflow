import hashlib
from typing import Optional, Any
import random
from serialize import MSGPackSerializer



class DHTID(int):
    HASH_FUNC = hashlib.sha1
    HASH_NBYTES = 20  # SHA1 produces a 20-byte (aka 160bit) number
    RANGE = MIN, MAX = 0, 2 ** (HASH_NBYTES * 8)  # inclusive min, exclusive max

    def __new__(cls, value: int):
        assert (
            cls.MIN <= value < cls.MAX
        ), f"DHTID must be in [{cls.MIN}, {cls.MAX}) but got {value}"
        return super().__new__(cls, value)

    @classmethod
    def generate(cls, source: Optional[Any] = None, nbits: int = 255):
        """
        Generates random uid based on SHA1

        :param source: if provided, converts this value to bytes and uses it as input for hashing function;
            by default, generates a random dhtid from :nbits: random bits
        """
        source = (
            random.getrandbits(nbits).to_bytes(nbits, byteorder="big")
            if source is None
            else source
        )
        source = (
            MSGPackSerializer.dumps(source) if not isinstance(source, bytes) else source
        )
        raw_uid = cls.HASH_FUNC(source).digest()
        return cls(int(raw_uid.hex(), 16))

    # def xor_distance(
    #     self, other: Union[DHTID, Sequence[DHTID]]
    # ) -> Union[int, List[int]]:
    #     """
    #     :param other: one or multiple DHTIDs. If given multiple DHTIDs as other, this function
    #      will compute distance from self to each of DHTIDs in other.
    #     :return: a number or a list of numbers whose binary representations equal bitwise xor between DHTIDs.
    #     """
    #     if isinstance(other, Iterable):
    #         return list(map(self.xor_distance, other))
    #     return int(self) ^ int(other)

    # @classmethod
    # def longest_common_prefix_length(cls, *ids: DHTID) -> int:
    #     ids_bits = [bin(uid)[2:].rjust(8 * cls.HASH_NBYTES, "0") for uid in ids]
    #     return len(os.path.commonprefix(ids_bits))

    # def to_bytes(self, length=HASH_NBYTES, byteorder="big", *, signed=False) -> bytes:
    #     """A standard way to serialize DHTID into bytes"""
    #     return super().to_bytes(length, byteorder, signed=signed)

    # @classmethod
    # def from_bytes(cls, raw: bytes, byteorder="big", *, signed=False) -> DHTID:
    #     """reverse of to_bytes"""
    #     return DHTID(super().from_bytes(raw, byteorder=byteorder, signed=signed))

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({hex(self)})"

    # def __bytes__(self):
    #     return self.to_bytes()


ret = DHTID.generate()
print(ret)  # 1088710293072759154972778073124457522945288211071

# If you have a DHT ID, then you are doomed to be identified in DHT!
