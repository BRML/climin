# -*- coding: utf-8 -*-

from datetime import datetime
import json
import uuid

from util import coroutine, aslist


def sane_to_json(item):
    """Return a copy of item where each array is replaced with a list."""
    if isinstance(item, dict):
        item = dict((k, sane_to_json(item[k])) for k in item)
    elif isinstance(item, list):
        item = [sane_to_json(i) for i in item]
    elif isinstance(item, tuple):
        item = tuple(sane_to_json(i) for i in item)
    elif hasattr(item, 'tolist'):
        item = item.tolist()

    return item


@coroutine
def jsonify(consumer):
    """Return a consumer which passes values on as json string."""
    while True:
        info = (yield)
        info = sane_to_json(info)
        consumer.send(json.dumps(info))


@coroutine
def print_sink():
    """Return a consumer that prints values received."""
    while True:
        info = (yield)
        print info


@coroutine
def prettyprint_sink():
    """Return a consumer that prints values received prettily."""
    while True:
        info = (yield)
        for key in info:
            print '%s: %s' % (key, info[key])
        print '-' * 20


@coroutine
def broadcast(*consumers):
    """Return a consumer that broadcasts values to all given consumers."""
    while True:
        info = (yield)
        for c in consumers:
            c.send(info)


@coroutine
def file_sink(filename, append=False, suffix='\n'):
    """Return a consumer that writes values to a file.

    If `append` is True, the file is opened for appending.
    Each value written to the files is followed by `suffix`."""
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        while True:
            info = (yield)
            f.write(str(info) + suffix)


@coroutine
def filelike_sink(file_like_object, suffix='\n'):
    """Return a consumer that writes values to a file like object.

    Each value written to the files is followed by `suffix`."""
    while True:
        info = (yield)
        file_like_object.write(str(info) + suffix)


@coroutine
def timify(consumer):
    """Returnt a consumer that adds a field 'datetime' to the received values
    and passes them on."""
    while True:
        info = (yield)
        info['datetime'] = datetime.now().isoformat()
        consumer.send(info)


@coroutine
def taggify(consumer, tags):
    """Return a consumer that adds a list of tags into the 'tags' field of the
    values and passes them on."""
    tags = aslist(tags)
    while True:
        info = (yield)
        info['tags'] = info.get('tags', [])
        info['tags'] += tags
        consumer.send(info)


@coroutine
def list_sink(lst):
    """Return a consumer that appends all received values to a list `lst`."""
    while True:
        info = (yield)
        lst.append(info)


@coroutine
def exclude_tags(consumer, tags):
    """Return a consumer that only passes values on that have none of a given 
    set of `tags`."""
    tags = aslist(tags)
    while True:
        info = (yield)
        if not 'tags' in info or all(i not in info['tags'] for i in tags):
            consumer.send(info)
        else:
            continue


@coroutine
def include_tags_only(consumer, tags):
    """Return a consumer that only passes values on that have all of a given 
    set of `tags`."""
    tags = aslist(tags)
    while True:
        info = (yield)
        if not 'tags' in info or not all(i in info['tags'] for i in tags):
            continue
        consumer.send(info)


@coroutine
def keep(consumer, keys):
    """Return consumer that only keeps a subset of the dictionary given by
    `keys`."""
    while True:
        info = (yield)
        new_info = dict((k, v) for k, v in info.items() if k in keys)
        consumer.send(new_info)


@coroutine
def dontkeep(consumer, keys):
    """Return consumer that throws away a subset of the dictionary given by
    `keys`."""
    while True:
        info = (yield)
        new_info = dict((k, v) for k, v in info.items() if k not in keys)
        consumer.send(new_info)


@coroutine
def deadend():
    """Return a consumer that does not do anything."""
    while True:
        (yield)


@coroutine
def uniquify(consumer):
    uid = str(uuid.uuid4())
    while True:
        info = (yield).copy()
        info['uuid'] = uid
        consumer.send(info)
