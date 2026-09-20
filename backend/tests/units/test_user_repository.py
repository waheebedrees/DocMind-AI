from sqlalchemy.ext.asyncio import AsyncSession
import pytest
from app.db.repositories.user_repo import UserRepository
from app.models.user import User


def make_user(
    email="ewaheeb02@gmail.com",
    password_hash='password_hash',
    is_active=True,
    is_verified=True,
):
    return User(
        email=email,
        password_hash=password_hash,
        is_active=is_active,
        is_verified=is_verified,
    )
    

@pytest.mark.unit
@pytest.mark.asyncio
class TestUserRepository:
    
    async def test_get_user_by_email(self, db: AsyncSession):
        repo = UserRepository(db)
        user = make_user()
        db.add(user)
        await db.flush()
        await db.refresh(user)
        res = await repo.get_by_email('ewaheeb02@gmail.com')
        assert res.email  is not None
        
    async def test_get_by_email_not_found(self, db:AsyncSession):
        repo = UserRepository(db)
        res = await repo.get_by_email('Not_found_email@email.com')
        assert res is None
        
    async def test_exist_by_email_not_false(self, db: AsyncSession):
        repo = UserRepository(db)
        res = await repo.exists_by_email('Not_found_email@email.com')
        assert res is False
        
    async def test_exist_by_email(self, db:AsyncSession):
        repo = UserRepository(db)
        
        user = make_user()

        db.add(user)
        await db.flush()
        await db.refresh(user)
        res = await repo.exists_by_email('ewaheeb02@gmail.com')
        assert res is True
        
    
    async def create_user(self, db:AsyncSession):
        repo = UserRepository(db)
        
        data = make_user()
        user = await repo.create_user(
            email=data.email,
            hash_password=data.password_hash,
            is_active=data.is_active,
            is_verified=data.is_verified
        )
        
        assert user.id is not None
        assert user.email == data.email
        assert user.is_active == data.is_active
        assert user.is_verified == data.is_verified
        