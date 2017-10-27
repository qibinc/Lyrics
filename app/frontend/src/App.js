import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';
import Grid from 'material-ui/Grid';
import Paper from 'material-ui/Paper';
import AppBar from 'material-ui/AppBar';
import Toolbar from 'material-ui/Toolbar';
import Typography from 'material-ui/Typography';
import Word from './Word';
import Seq2Seq from './Seq2Seq';

const styles = theme => ({
    root: {
        flexGrow: 1,
        textAlign: "center",
        padding: theme.spacing.unit * 3,
    },
    flex: {
        flex: 1,
    },
    paper: {
        textAlign: 'center',
        color: theme.palette.text.secondary,
        marginTop: theme.spacing.unit * 3,
    },
});

/**
 * The app which consists of ...
 */
class App extends Component
{
    render() {
        const { classes } = this.props;

        return (
            <div className={classes.root}>
                <AppBar className={classes.appbar} position="static" >
                    <Toolbar>
                        <Typography className={classes.flex} type="title" color="inherit">
                            Deep Lyrics
                        </Typography>
                    </Toolbar>
                </AppBar>
                <Grid container className={classes.container}>
                    <Grid item xs={12} sm={12}>
                        <Paper className={classes.paper}>
                            <Word/>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} sm={12}>
                        <Paper className={classes.paper}>
                            <Seq2Seq/>
                        </Paper>
                    </Grid>
                </Grid>
            </div>
        );
    }
}

App.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(App);

