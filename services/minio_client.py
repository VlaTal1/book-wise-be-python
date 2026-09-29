"""
Спільний MinIO-клієнт для сервісів, яким потрібен прямий доступ до бакету
книг (той самий бакет, що й у Java — settings.minio_bucket/MinioService.
bucketName мають збігатися). Раніше кожен модуль (test_generator.py,
services/book_excerpt.py) створював свій власний Minio(...) з тими самими
параметрами — тепер, коли з'явився третій випадок використання (завантаження
аудіо сесії читання для історії), винесено сюди, щоб не потроювати
дублювання. test_generator.py навмисно НЕ чіпали — не пов'язаний з цією
задачею функціонал, ризик регресії не виправдовує рефакторинг заради нього.
"""

from minio import Minio

from core.config import settings

minio_client = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure,
)
