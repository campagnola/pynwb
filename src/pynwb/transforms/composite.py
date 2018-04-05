# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See vispy/LICENSE.txt for more info.

from __future__ import division

from .base_transform import BaseTransform
from .linear import NullTransform


class CompositeTransform(BaseTransform):
    """
    BaseTransform subclass that performs a sequence of transformations in
    order.

    Arguments:

    transforms : list of BaseTransform instances
        See ``transforms`` property.
    """
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, *transforms):
        super(CompositeTransform, self).__init__()
        self._transforms = []
        self._simplified = None
        self._null_transform = NullTransform()
        
        # Set input transforms
        trs = []
        for tr in transforms:
            if isinstance(tr, (tuple, list)):
                trs.extend(tr)
            else:
                trs.append(tr)
        self.transforms = trs

    @property
    def transforms(self):
        """ The list of transform that make up the transform chain.
        
        The order of transforms is given such that the last transform in the 
        list is the first to be invoked when mapping coordinates through 
        the chain. 
        
        For example, the following two mappings are equivalent::
        
            # Map coordinates through individual transforms:
            trans1 = STTransform(scale=(2, 3), translate=(0, 1))
            trans2 = PolarTransform()
            mapped = trans1.map(trans2.map(coords))
            
            # Equivalent mapping through chain:
            chain = CompositeTransform([trans1, trans2])
            mapped = chain.map(coords)
            
        """
        return self._transforms

    @transforms.setter
    def transforms(self, tr):
        if isinstance(tr, BaseTransform):
            tr = [tr]
        if not isinstance(tr, list):
            raise TypeError("Transform chain must be a list")
        
        # Avoid extra effort if we already have the correct chain
        if len(tr) == len(self._transforms):
            changed = False
            for i in range(len(tr)):
                if tr[i] is not self._transforms[i]:
                    changed = True
                    break
            if not changed:
                return
        
        for t in self._transforms:
            t.remove_change_callback(self._subtr_changed)
        self._transforms = tr
        for t in self._transforms:
            t.add_change_callback(self._subtr_changed)
        self.update()

    @property
    def simplified(self):
        """A simplified representation of the same transformation.
        """
        if self._simplified is None:
            self._simplified = SimplifiedCompositeTransform(self)
        return self._simplified

    @property
    def Linear(self):
        b = True
        for tr in self._transforms:
            b &= tr.Linear
        return b

    @property
    def Orthogonal(self):
        b = True
        for tr in self._transforms:
            b &= tr.Orthogonal
        return b

    @property
    def NonScaling(self):
        b = True
        for tr in self._transforms:
            b &= tr.NonScaling
        return b

    @property
    def Isometric(self):
        b = True
        for tr in self._transforms:
            b &= tr.Isometric
        return b

    def map(self, coords):
        """Map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        for tr in reversed(self.transforms):
            coords = tr.map(coords)
        return coords

    def imap(self, coords):
        """Inverse map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        for tr in self.transforms:
            coords = tr.imap(coords)
        return coords

    def append(self, tr):
        """
        Add a new transform to the end of this chain.

        Parameters
        ----------
        tr : instance of Transform
            The transform to use.
        """
        self.transforms.append(tr)
        tr.add_change_callback(self._subtr_changed)
        self.update()

    def prepend(self, tr):
        """
        Add a new transform to the beginning of this chain.

        Parameters
        ----------
        tr : instance of Transform
            The transform to use.
        """
        self.transforms.insert(0, tr)
        tr.add_change_callback(self._subtr_changed)
        self.update()

    def _subtr_changed(self, ev):
        """One of the internal transforms changed; propagate the signal. 
        """
        self.update(ev)

    def __setitem__(self, index, tr):
        self._transforms[index].remove_change_callback(self._subtr_changed)
        self._transforms[index] = tr
        tr.add_change_callback(self.subtr_changed)
        self.update()

    def __mul__(self, tr):
        if isinstance(tr, CompositeTransform):
            trs = tr.transforms
        else:
            trs = [tr]
        return CompositeTransform(self.transforms+trs)

    def __rmul__(self, tr):
        if isinstance(tr, CompositeTransform):
            trs = tr.transforms
        else:
            trs = [tr]
        return CompositeTransform(trs+self.transforms)

    def __str__(self):
        names = [tr.__class__.__name__ for tr in self.transforms]
        return "<CompositeTransform [%s] at 0x%x>" % (", ".join(names), id(self))
    
    def __repr__(self):
        tr = ",\n                 ".join(map(repr, self.transforms))
        return "<CompositeTransform [%s] at 0x%x>" % (tr, id(self))


class SimplifiedCompositeTransform(CompositeTransform):
    def __init__(self, chain):
        CompositeTransform.__init__(self)
        self._chain = chain
        chain.add_change_callback(self.source_changed)
        self.source_changed(None)

    def source_changed(self, event):
        """Generate a simplified chain by joining adjacent transforms.
        """
        # bail out early if the chain is empty
        transforms = self._chain.transforms[:]
        if len(transforms) == 0:
            self.transforms = []
            return
        
        # If the change signal comes from a transform that already appears in
        # our simplified transform list, then there is no need to re-simplify.
        if event is not None:
            for source in event.sources[::-1]:
                if source in self.transforms:
                    self.update(event)
                    return
        
        # First flatten the chain by expanding all nested chains
        new_chain = []
        while len(transforms) > 0:
            tr = transforms.pop(0)
            if isinstance(tr, CompositeTransform) and not tr.dynamic:
                transforms = tr.transforms[:] + transforms
            else:
                new_chain.append(tr)
        
        # Now combine together all compatible adjacent transforms
        cont = True
        tr = new_chain
        while cont:
            new_tr = [tr[0]]
            cont = False
            for t2 in tr[1:]:
                t1 = new_tr[-1]
                pr = t1 * t2
                if (not t1.dynamic and not t2.dynamic and not 
                   isinstance(pr, CompositeTransform)):
                    cont = True
                    new_tr.pop()
                    new_tr.append(pr)
                else:
                    new_tr.append(t2)
            tr = new_tr

        self.transforms = tr