import cStringIO
import json
import tempfile
import os

import climin.logging as L


def test_timing():
    lst = []
    pipe = L.list_sink(lst)
    pipe = L.timify(pipe)
    pipe.send({})
    assert len(lst) == 1, 'nothing added to sink'
    assert 'datetime' in lst[0]


def test_tagging():
    lst = []
    pipe = L.list_sink(lst)
    pipe = L.taggify(pipe, tags=['bla', 'blubb'])
    pipe.send({})
    assert len(lst) == 1, 'nothing added to sink'
    tags = lst[0]['tags']
    assert tags == ['bla', 'blubb'], 'tags did not get through: %s' % tags

    pipe = L.taggify(pipe, tags='hopp')
    pipe.send({})
    tags = lst[-1]['tags']
    assert tags == ['hopp', 'bla', 'blubb'], 'tags did not get through: %s' % tags


def test_jsonify():
    lst = []
    pipe = L.list_sink(lst)
    pipe = L.jsonify(pipe)
    info = {'2': [3], '4': 5.}
    pipe.send(info)
    received_info = json.loads(lst[0])
    assert received_info == info, 'jsonify did mutate info: %s' % received_info


def test_broadcast():
    lst1 = []
    lst2 = []
    pipe1 = L.list_sink(lst1)
    pipe2 = L.list_sink(lst2)
    pipe = L.broadcast(pipe1, pipe2)
    info = {'und alle so': 'yeah'}
    pipe.send(info)
    assert lst1[0] == info, 'list 1 did not receive info'
    assert lst2[0] == info, 'list 2 did not receive info'


def test_filelike_sink():
    flo = cStringIO.StringIO()
    info = {'und alle so': 'yeah'}
    pipe = L.filelike_sink(flo, '')
    pipe.send(info)
    assert flo.getvalue() == str(info)

    flo = cStringIO.StringIO()
    info = {'und alle so': 'yeah'}
    pipe = L.filelike_sink(flo)
    pipe.send(info)
    assert flo.getvalue() == str(info) + '\n'


def test_file_sink():
    info = {'und alle so': 'yeah'}

    # 1. Test without newline.

    # Create a file on the file system.
    tf = tempfile.NamedTemporaryFile('r', delete=False)
    tf.close()

    # Send info into file.
    pipe = L.file_sink(tf.name, suffix='')
    pipe.send(info)
    del pipe

    with open(tf.name) as fp:
        content = fp.read()
    assert content == str(info)
    # Delete file.
    os.remove(tf.name)

    # 2. Test with newline.
    # Create a file on the file system.
    tf = tempfile.NamedTemporaryFile('r', delete=False)
    tf.close()

    # Send info into file.
    pipe = L.file_sink(tf.name, suffix='\n')
    pipe.send(info)
    del pipe

    with open(tf.name) as fp:
        content = fp.read()

    assert content == str(info) + '\n'

    # Send to it again.
    pipe = L.file_sink(tf.name, append=True, suffix='\n')
    pipe.send(info)
    del pipe

    with open(tf.name) as fp:
        content = fp.read()

    assert content == str(info) + '\n' + str(info) + '\n', 'content not as expected'


    # Delete file.
    os.remove(tf.name)


def test_exclude_tags():
    lst = []
    pipe = L.list_sink(lst)
    pipe = L.exclude_tags(pipe, 'fun')
    pipe1 = L.taggify(pipe, 'warn')
    pipe2 = L.taggify(pipe, 'fun')

    pipe1.send({'message': 'take care'})
    pipe2.send({'message': 'haahaa'})

    assert len(lst) == 1, 'wrong number of messages got through'
    assert lst[0]['message'] == 'take care', 'wrong message got through'


def test_project():
    lst = []
    pipe = L.list_sink(lst)
    pipe = L.project(pipe, ['message'])

    pipe.send({'message': 'take care', 'blabla': 'blubb'})

    assert len(lst) == 1, 'wrong number of messages got through'
    assert lst[0] == {'message': 'take care'}, 'wrong keys got through'

def test_uniquify():
    lst = []
    pipe = L.list_sink(lst)
    pipe = L.uniquify(pipe)

    pipe.send({'message': 'take care', 'blabla': 'blubb'})

    assert len(lst) == 1, 'wrong number of messages got through'
    assert 'uuid' in lst[0], 'did not get a uuid'
    assert len(lst[0]['uuid']) == 36, 'not a uuid'
