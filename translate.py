from deviz.lang.stmt import (
    BaseStmt,
    CalculateOp,
    CalculateStmt,
    Filter,
    FilterOp,
    FilterStmt,
    JoinStmt,
    RankStmt,
    SelectStmt,
    SummarizeOp,
    SummarizeStmt,
)
from deviz.lang.data import SymColumn, SymTable

import pandas as pd

import os
import json


class Col:
    def __init__(self, initial_columns):
        self.initial_columns = initial_columns
    
    def __getattribute__(self, name: str):
        return Attr(name, self.initial_columns)
    
    def __getitem__(self, name: str):
        assert isinstance(name, str)
        return Attr(name, self.initial_columns)


class Attr:
    def __init__(self, name, initial_columns):
        self.name = name
        self.initial_columns = initial_columns

    def __eq__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.EQ,
            value=other,
        )
    
    def __req__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.EQ,
            value=other,
        )

    def __ne__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.NEQ,
            value=other,
        )
    
    def __rne__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.NEQ,
            value=other,
        )

    def __gt__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.GT,
            value=other,
        )
    
    def __rgt__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.LT,
            value=other,
        )

    def __ge__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.GTE,
            value=other,
        )
    
    def __rge__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.LTE,
            value=other,
        )

    def __lt__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.LT,
            value=other,
        )
    
    def __rlt__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.GT,
            value=other,
        )

    def __le__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.LTE,
            value=other,
        )
    
    def __rle__(self, other):
        assert not isinstance(other, Attr)
        return Filter(
            column=self.initial_columns[self.name],
            op=FilterOp.GTE,
            value=other,
        )
    
    def __add__(self, other):
        assert isinstance(other, Attr)
        def calculate_stmt(table, output_table, new_column):
            return CalculateStmt(
                table=table,
                output_table=output_table,
                lhs_column=self.initial_columns[self.name],
                op=CalculateOp.ADD,
                rhs_column=self.initial_columns[other.name],
                new_column=new_column,
            )
        return calculate_stmt


def vlfilter(
    transform,
    tables: 'list[SymTable]',
    initial_columns: 'dict[str, SymColumn]',
):
    tn = SymTable(df=None, identifier=f"t{len(tables)}")
    
    filter_stmt = FilterStmt(
        table=tables[-1],
        output_table=tn,
        filters=[
            (eval(f'(lambda datum: {f})'))(Col(initial_columns))
            for f
            in transform['filter'].split('&&')
        ]
    )
    filter_stmt.concretize()
    return [tn], [filter_stmt]


def summarize_one(agg, groupby: list[str], in_table: "SymTable", out_table: "SymTable", initial_columns: 'dict[str, SymColumn]'):
    # Todo: how about output column name? (agg['as'])
    summarize_stmt = SummarizeStmt(
        table=in_table,
        output_table=out_table,
        group_by_columns=[initial_columns[col] for col in groupby],
        # Todo: which agg_colukn to use when "counting"
        agg_column=initial_columns[agg['field'] or [*initial_columns.keys()][0]],
        agg_op=SummarizeOp(agg['op']),
    )

    summarize_stmt.concretize()
    return summarize_stmt


def vlaggregate(
    transform,
    tables: 'list[SymTable]',
    initial_columns: 'dict[str, SymColumn]',
):
    aggs = transform['aggregate']
    groupby: list[str] = transform['groupby']

    if len(aggs) == 1:
        tn = SymTable(df=None, identifier=f"t{len(tables)}")
        summarize_stmt = summarize_one(aggs[0], groupby, tables[-1], tn, initial_columns)
        return [tn], [summarize_stmt]
    else:
        tns = [SymTable(df=None, identifier=f"t{len(tables) + i}") for i in range(len(aggs))]
        summarize_stmts: 'list[SummarizeStmt]' = []
        for tn, agg in zip(tns, aggs):
            summarize_stmt = summarize_one(agg, groupby, tables[-1], tn, initial_columns)
            summarize_stmts.append(summarize_stmt)

        tn = SymTable(df=None, identifier=f"t{len(tables) + len(tns)}")
        join_stmt = JoinStmt(
            table=tns,
            output_table=tn
        )

        join_stmt.concretize()
        return tns + [tn], summarize_stmts + [join_stmt]


def vlrank(
    transform,
    tables: 'list[SymTable]',
    initial_columns: 'dict[str, SymColumn]',
):
    windows = transform['window']
    groupby: list[str] = transform['groupby']
    sorts = transform['sort']

    assert len(windows) == 1, windows
    window = windows[0]

    assert len(sorts) == 1, sorts
    sort = sorts[0]

    assert 'frame' not in transform
    assert 'ignorePeers' not in transform

    assert window['op'] == 'rank', window

    tn = SymTable(df=None, identifier=f"t{len(tables)}")
    rank_stmt = RankStmt(
        table=tables[-1],
        output_table=tn,
        # Todo: how about sorting order? (asc / desc) from sort['order']
        measure=initial_columns[sort['field']],
        partition_columns=[initial_columns[col] for col in groupby],
        new_column=SymColumn(column_name=window['as'], dtype='int64', label=window['as'])
    )
    return [tn], [rank_stmt]


def vlcalculate(
    transform,
    tables: 'list[SymTable]',
    initial_columns: 'dict[str, SymColumn]',
):
    calculate = transform['calculate']
    column_name = transform['as']

    tn = SymTable(df=None, identifier=f"t{len(tables)}")
    
    calculate_stmt = (eval(f'(lambda datum: {calculate})'))(Col(initial_columns))(
        tables[-1],
        tn,
        SymColumn(
            column_name=column_name,
            dtype="object",
            label=column_name,
        ),
    )
    calculate_stmt.concretize()
    return [tn], [calculate_stmt]
    


def main():
    with open('./nlvcorpus/vlSpecs.json', 'r') as f:
        specs = json.load(f)
    
    for k, v in specs.items():
        try:
            data = v['data']
            dataUrl = data['url']
            print(k, dataUrl)


            df = pd.read_csv(os.path.join('nlvcorpus', dataUrl))
            table = SymTable(df=df, identifier=k)
            initial_columns = table.get_sym_columns()

            tables: 'list[SymTable]' = [table]
            stmts: 'list[BaseStmt]' = []

            transforms = v.get('transforms', [])

            encoding = v['encoding']
            x = encoding['x']
            y = encoding['y']
            color = encoding.get('color', None)
            facet = encoding.get('facet', None)

            channels = [ch for ch in (x, y, color, facet) if ch is not None]
            if any('aggregate' in ch for ch in channels):
                transforms.append({
                    "aggregate": [{
                        "op": ch['aggregate'],
                        "field": ch.get('field', None),
                        # Todo: Should rename
                        "as": ch.get('field', ch['aggregate']),
                    } for ch in channels if 'aggregate' in ch],
                    "groupby": [*set(ch['field'] for ch in channels if 'aggregate' not in ch)],
                })

            for t in transforms:
                if "filter" in t:
                    tn, stmt = vlfilter(t, tables, initial_columns)
                elif "aggregate" in t:
                    tn, stmt = vlaggregate(t, tables, initial_columns)
                elif "calculate" in t:
                    tn, stmt = vlcalculate(t, tables, initial_columns)
                elif "window" in t:
                    tn, stmt = vlrank(t, tables, initial_columns)
                else:
                    raise Exception('Transformation not supported:', json.dumps(t))
                tables.extend(tn)
                stmts.extend(stmt)
                # for table in tables:
                #     print("  ", table)
                for stmt in stmts:
                    print("     ", stmt)
        except Exception as e:
            # print(e)
            print(json.dumps(v, indent=2))
            raise e


if __name__ == "__main__":
    main()