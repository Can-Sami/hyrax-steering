from dataclasses import dataclass


@dataclass(frozen=True)
class SeedIntent:
    intent_code: str
    description: str
    utterances: tuple[str, ...]


TURKISH_INTENT_SEED: tuple[SeedIntent, ...] = (
    SeedIntent(
        intent_code='fatura_sorgulama',
        description='Kullanici fatura bilgilerini ogrenmek istiyor.',
        utterances=('Faturam ne kadar?', 'Son faturamı öğrenmek istiyorum.'),
    ),
    SeedIntent(
        intent_code='tarife_degistirme',
        description='Kullanici mevcut tarife/paketini degistirmek istiyor.',
        utterances=('Tarifemi değiştirmek istiyorum.', 'Daha uygun paket var mı?'),
    ),
)
