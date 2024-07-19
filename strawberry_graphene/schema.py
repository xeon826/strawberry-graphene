import datetime
import inspect
from decimal import Decimal
from enum import Enum
from functools import partial
from graphene.utils.str_converters import to_snake_case
from typing import Any, Dict, Optional, Sequence, Type, Union

import graphene
import strawberry
from graphene.types.base import BaseType as BaseGrapheneType
from graphene.types.definitions import GrapheneUnionType
from graphene.types.schema import TypeMap as BaseGrapheneTypeMap
from graphql import (
    ExecutionContext as GraphQLExecutionContext,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLType,
    validate_schema,
)
from graphql.type.directives import specified_directives
from strawberry.custom_scalar import ScalarDefinition, ScalarWrapper
from strawberry.directive import StrawberryDirective
from strawberry.enum import EnumDefinition
from strawberry.extensions import Extension
from strawberry.field import StrawberryField
from strawberry.schema import schema_converter
from strawberry.schema.config import StrawberryConfig
from strawberry.schema.schema_converter import CustomGraphQLEnumType
from strawberry.schema.types import ConcreteType
from strawberry.schema.types.scalar import DEFAULT_SCALAR_REGISTRY
from strawberry.types.types import TypeDefinition
from strawberry.union import StrawberryUnion
from strawberry.utils.str_converters import to_camel_case


def to_enum_name(s: str) -> str:
    return to_snake_case(s.replace('.', '_').replace('-', '_')).upper()


class OurCustomGraphQLEnumType(CustomGraphQLEnumType):
    def serialize(self, output_value: Any) -> str:
        if isinstance(output_value, Enum):
            return to_enum_name(output_value.value)
        return super().serialize(output_value)


class GraphQLCoreConverter(schema_converter.GraphQLCoreConverter):
    def __init__(self, *args, **kwargs):
        self.graphene_type_map = GrapheneTypeMap(self)
        super().__init__(*args, **kwargs)

    def add_graphene_type(self, type_: Any) -> GraphQLObjectType:
        return self.graphene_type_map.add_type(type_)

    def from_object_type(self, object_type: Type) -> GraphQLObjectType:
        # Check if it's a Graphene type
        if issubclass(object_type, graphene.ObjectType):
            return self.add_graphene_type(object_type)

        return self.from_object(object_type._type_definition)

    def from_type(self, type_: Any) -> GraphQLType:
        if inspect.isclass(type_) and issubclass(type_, BaseGrapheneType):
            return self.add_graphene_type(type_)
        return super().from_type(type_)

    def from_enum(self, enum: EnumDefinition) -> CustomGraphQLEnumType:
        enum_name = self.config.name_converter.from_type(enum)

        assert enum_name is not None

        # Don't reevaluate known types
        if enum_name in self.type_map:
            graphql_enum = self.type_map[enum_name].implementation
            assert isinstance(
                graphql_enum, OurCustomGraphQLEnumType
            )  # For mypy
            return graphql_enum

        graphql_enum = OurCustomGraphQLEnumType(
            name=enum_name,
            values={
                to_enum_name(item.value): self.from_enum_value(item)
                for item in enum.values
            },
            description=enum.description,
        )

        self.type_map[enum_name] = ConcreteType(
            definition=enum, implementation=graphql_enum
        )

        return graphql_enum


class GrapheneTypeMap(BaseGrapheneTypeMap):
    def __init__(self, strawberry_convertor, *args, **kwargs):
        self.strawberry_convertor = strawberry_convertor
        super().__init__(*args, **kwargs)

    def construct_union(self, graphene_type):
        create_graphql_type = self.add_type

        def types():
            union_types = []
            for graphene_objecttype in graphene_type._meta.types:
                object_type = create_graphql_type(graphene_objecttype)
                union_types.append(object_type)
            return union_types

        resolve_type = (
            partial(
                self.resolve_type,
                graphene_type.resolve_type,
                graphene_type._meta.name,
            )
            if graphene_type.resolve_type
            else None
        )

        return GrapheneUnionType(
            graphene_type=graphene_type,
            name=graphene_type._meta.name,
            description=graphene_type._meta.description,
            types=types,
            resolve_type=resolve_type,
        )

    def add_type(self, graphene_type):
        if hasattr(graphene_type, "_type_definition") or hasattr(
            graphene_type, "_enum_definition"
        ):
            try:
                return self.strawberry_convertor.from_type(graphene_type)
            except RecursionError as e:
                raise RuntimeError(
                    f'You probably have @strawberry.type decorator on a class that inherits from graphene. Please remove the graphene base class from {graphene_type}.'
                ) from e

        # Special case decimal
        if isinstance(graphene_type, type) and issubclass(
            graphene_type, graphene.Decimal
        ):
            return self.strawberry_convertor.from_scalar(Decimal)
        if isinstance(graphene_type, type) and issubclass(
            graphene_type, graphene.DateTime
        ):
            return self.strawberry_convertor.from_scalar(datetime.datetime)

        if inspect.isfunction(graphene_type):
            graphene_type = graphene_type()
        if isinstance(graphene_type, graphene.List):
            return GraphQLList(self.add_type(graphene_type.of_type))
        if isinstance(graphene_type, graphene.NonNull):
            return GraphQLNonNull(self.add_type(graphene_type.of_type))
        try:
            name = graphene_type._meta.name
        except AttributeError as e:
            raise TypeError(
                f"Expected Graphene type, but received: {graphene_type}."
            ) from e
        graphql_type = self.get(name)
        if graphql_type:
            return graphql_type
        if issubclass(graphene_type, graphene.ObjectType):
            graphql_type = self.create_objecttype(graphene_type)

            # Calling vars(graphene_type).items() on some graphene objects
            # causes later infinite recursion.. Quick hack to get moving for
            # now.
            if hasattr(graphene_type, 'has_strawberry_field'):
                # Create gql fields for all strawberry fields/mutations,
                # because create_objecttype(graphene_type) creates only
                # graphene fields.
                # This attr must be set on all graphene classes that
                # reference strawberry field or mutation.
                for name, field in vars(graphene_type).items():
                    if isinstance(field, StrawberryField):
                        gql_field = self.strawberry_convertor.from_field(field)
                        camel_case_name = to_camel_case(name)
                        graphql_type.fields[camel_case_name] = gql_field

        elif issubclass(graphene_type, graphene.InputObjectType):
            graphql_type = self.create_inputobjecttype(graphene_type)
        elif issubclass(graphene_type, graphene.Interface):
            # allows us to prefer the strawberry implementation of an interface
            # (mostly useful for Node)
            if graphene_type._meta.name in self.strawberry_convertor.type_map:
                graphql_type = self.strawberry_convertor.type_map[
                    graphene_type._meta.name
                ].implementation
                graphql_type.graphene_type = graphene_type
            else:
                graphql_type = self.create_interface(graphene_type)
        elif issubclass(graphene_type, graphene.Scalar):
            graphql_type = self.create_scalar(graphene_type)
        elif issubclass(graphene_type, graphene.Enum):
            graphql_type = self.create_enum(graphene_type)
        elif issubclass(graphene_type, graphene.Union):
            graphql_type = self.construct_union(graphene_type)
        else:
            raise TypeError(
                f"Expected Graphene type, but received: {graphene_type}."
            )
        self[name] = graphql_type
        if not issubclass(graphene_type, graphene.Scalar):
            self.strawberry_convertor.type_map[name] = ConcreteType(
                definition=None, implementation=graphql_type
            )
        return graphql_type


class Schema(strawberry.Schema):
    def __init__(
        self,
        # TODO: can we make sure we only allow to pass something that has been decorated?
        query: Type,
        mutation: Optional[Type] = None,
        subscription: Optional[Type] = None,
        directives: Sequence[StrawberryDirective] = (),
        types=(),
        extensions: Sequence[Union[Type[Extension], Extension]] = (),
        execution_context_class: Optional[
            Type[GraphQLExecutionContext]
        ] = None,
        config: Optional[StrawberryConfig] = None,
        scalar_overrides: Optional[
            Dict[object, Union[ScalarWrapper, ScalarDefinition]]
        ] = None,
    ):
        self.extensions = extensions
        self.execution_context_class = execution_context_class
        self.config = config or StrawberryConfig()

        scalar_registry: Dict[
            object, Union[ScalarWrapper, ScalarDefinition]
        ] = {**DEFAULT_SCALAR_REGISTRY}
        if scalar_overrides:
            scalar_registry.update(scalar_overrides)

        self.schema_converter = GraphQLCoreConverter(
            self.config, scalar_registry
        )
        self.directives = directives

        query_type = self.schema_converter.from_object_type(query)
        mutation_type = (
            self.schema_converter.from_object_type(mutation)
            if mutation
            else None
        )
        subscription_type = (
            self.schema_converter.from_object_type(subscription)
            if subscription
            else None
        )

        graphql_directives = [
            self.schema_converter.from_directive(directive)
            for directive in directives
        ]

        graphql_types = []
        for type_ in types:
            graphql_type = self.schema_converter.from_object_type(type_)
            graphql_types.append(graphql_type)

        self._schema = GraphQLSchema(
            query=query_type,
            mutation=mutation_type,
            subscription=subscription_type if subscription else None,
            directives=specified_directives + graphql_directives,
            types=graphql_types,
        )

        # attach our schema to the GraphQL schema instance
        self._schema._strawberry_schema = self  # type: ignore

        # Validate schema early because we want developers to know about
        # possible issues as soon as possible
        errors = validate_schema(self._schema)
        if errors:
            formatted_errors = "\n\n".join(
                f"❌ {error.message}" for error in errors
            )
            raise ValueError(f"Invalid Schema. Errors:\n\n{formatted_errors}")

        self.query = self.schema_converter.type_map[query_type.name]

    def get_type_by_name(
        self, name: str
    ) -> Optional[
        Union[
            TypeDefinition, ScalarDefinition, EnumDefinition, StrawberryUnion
        ]
    ]:
        if name in self.schema_converter.type_map:
            return getattr(
                self.schema_converter.type_map[name], "definition", None
            )

        return None

