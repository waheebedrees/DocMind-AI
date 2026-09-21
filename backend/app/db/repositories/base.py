from typing import TypeVar
from uuid import UUID

from sqlalchemy import select, update, Select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository[ModelType: Base]:
    """
    Generic repository for common CRUD operations
    provide clean interface for data access operations
    """

    def __init__(self, model: type[ModelType], session: AsyncSession):
        """initialize repository wih model and session

        Args:
            model SQLALchemy ORM model class
            session (AsyncSession): database session
        """
        self.session = session
        self.model = model

    async def get_by_id(self, id: UUID) -> ModelType | None:
        """
        get single recode by id

        Args:
            id (UUID): primary key

        Returns:
            Model instance or None if not found
        """
        stmt = select(self.model).where(self.model.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
        """
        get all all recode with pagination

        Args:
            skip (int, optional): number of recodes to skip. Defaults to 0.
            limit (int, optional): maximum number of recodes. Defaults to 100.

        Returns:
            List[ModelType]: list of model instance
        """
        stmt = select(self.model).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, **kwargs) -> ModelType:
        """
        create new recode

        Args:
            **kwargs: model attributes

        Returns:
            ModelType: created model instance

        Raises:
            IntegrityError: if unique constraint violation occurs
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.commit()
        await self.session.refresh(instance)
        return instance

    async def update(self, id: UUID, **kwargs) -> ModelType | None:
        """
        update existing recode

        Args:
            id (UUID): primary key
            **kwargs: model attributes to update

        Returns:
            ModelType | None: _description_
        """
        stmt = update(self.model).where(self.model.id == id).values(**kwargs).returning(self.model)

        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def count(self) -> int:
        """
        count total number of recodes

        Returns:
            int: total count
        """
        stmt = select(self.model)
        result = await self.session.execute(stmt)
        return len(result.scalars().all())

    async def exists(self, id: UUID) -> bool:
        """
        check if recode exists by id

        Args:
            id (UUID): primary key

        Returns:
            bool: True if exists False otherwise
        """
        instance = await self.get_by_id(id)
        return instance is not None

    async def delete(self, id: UUID) -> bool:

        instance = await self.get_by_id(id)

        if instance:
            await self.session.delete(instance)
            await self.session.flush()
            return True
        return False

    def page(self, stmt: Select, *, limit: int, offset: int) -> Select:
        return stmt.limit(limit).offset(offset)

    async def total(self, stmt: Select) -> int:
        """Count rows matching a SELECT, ignoring ORDER BY / LIMIT / OFFSET."""
        subquery = stmt.order_by(None).subquery()
        result = await self.session.scalar(select(func.count()).select_from(subquery))
        return int(result or 0)
